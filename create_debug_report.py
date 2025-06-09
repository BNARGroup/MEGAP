import os
import sys
import platform
import subprocess
import datetime
import json
import socket
import getpass
import pkg_resources
import psutil
import cpuinfo
from pathlib import Path

def get_system_info():
    """Collect general system information"""
    info = {}
    
    # Basic system info
    info['MEG_device']= str(os.getenv('MEGDEVICE'))
    info['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info['os_name'] = os.name
    info['platform_system'] = platform.system()
    info['platform_release'] = platform.release()
    info['platform_version'] = platform.version()
    info['platform_machine'] = platform.machine()
    info['platform_processor'] = platform.processor()
    info['architecture'] = platform.architecture()
    info['hostname'] = socket.gethostname()
    info['username'] = getpass.getuser()
    
    return info

def get_python_info():
    """Collect Python environment information"""
    info = {}
    
    info['python_version'] = sys.version
    info['python_executable'] = sys.executable
    info['python_path'] = sys.path
    info['python_prefix'] = sys.prefix
    info['python_base_prefix'] = getattr(sys, 'base_prefix', sys.prefix)
    info['is_virtual_env'] = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    
    # Virtual environment detection
    if 'VIRTUAL_ENV' in os.environ:
        info['virtual_env_path'] = os.environ['VIRTUAL_ENV']
        info['virtual_env_name'] = os.path.basename(os.environ['VIRTUAL_ENV'])
    elif 'CONDA_DEFAULT_ENV' in os.environ:
        info['conda_env'] = os.environ['CONDA_DEFAULT_ENV']
    
    return info

def get_hardware_info():
    """Collect detailed hardware information"""
    info = {}
    
    try:
        # CPU information
        cpu_info = cpuinfo.get_cpu_info()
        info['cpu_brand'] = cpu_info.get('brand_raw', 'Unknown')
        info['cpu_arch'] = cpu_info.get('arch', 'Unknown')
        info['cpu_bits'] = cpu_info.get('bits', 'Unknown')
        info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        info['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        
        # Memory information
        memory = psutil.virtual_memory()
        info['memory_total'] = f"{memory.total / (1024**3):.2f} GB"
        info['memory_available'] = f"{memory.available / (1024**3):.2f} GB"
        info['memory_percent'] = f"{memory.percent}%"
        
        # Disk information
        disk_usage = psutil.disk_usage('/')
        info['disk_total'] = f"{disk_usage.total / (1024**3):.2f} GB"
        info['disk_used'] = f"{disk_usage.used / (1024**3):.2f} GB"
        info['disk_free'] = f"{disk_usage.free / (1024**3):.2f} GB"
        info['disk_percent'] = f"{(disk_usage.used / disk_usage.total) * 100:.1f}%"
        
        # Boot time
        info['boot_time'] = datetime.datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        info['hardware_error'] = f"Error collecting hardware info: {str(e)}"
    
    return info

def get_installed_packages():
    """Get list of installed Python packages"""
    packages = {}
    
    try:
        # Using pkg_resources
        installed_packages = [d for d in pkg_resources.working_set]
        packages['pip_packages'] = {pkg.project_name: pkg.version for pkg in installed_packages}
        packages['total_packages'] = len(installed_packages)
    except Exception as e:
        packages['packages_error'] = f"Error getting packages: {str(e)}"
    
    # Try pip freeze as backup
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            packages['pip_freeze'] = result.stdout.strip().split('\n')
    except Exception as e:
        packages['pip_freeze_error'] = f"Error running pip freeze: {str(e)}"
    
    return packages

def get_environment_variables():
    """Get environment variables (filtered for security)"""
    # Sensitive patterns to exclude
    sensitive_patterns = [
        'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'CREDENTIAL', 
        'AUTH', 'PRIVATE', 'CERT', 'SSH', 'AWS', 'API'
    ]
    
    filtered_env = {}
    for key, value in os.environ.items():
        # Check if key contains sensitive patterns
        is_sensitive = any(pattern in key.upper() for pattern in sensitive_patterns)
        
        if is_sensitive:
            filtered_env[key] = "[REDACTED FOR SECURITY]"
        else:
            filtered_env[key] = value
    
    return filtered_env

def get_network_info():
    """Get basic network information"""
    info = {}
    
    try:
        info['hostname'] = socket.gethostname()
        info['fqdn'] = socket.getfqdn()
        
        # Get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            info['local_ip'] = s.getsockname()[0]
            
    except Exception as e:
        info['network_error'] = f"Error getting network info: {str(e)}"
    
    return info

def generate_debug_report():
    """Generate comprehensive debug report"""
    report = {
        'report_info': {
            'generated_at': datetime.datetime.now().isoformat(),
            'script_version': '1.0',
            'purpose': 'System debugging and troubleshooting'
        },
        'system_info': get_system_info(),
        'python_info': get_python_info(),
        'hardware_info': get_hardware_info(),
        'installed_packages': get_installed_packages(),
        'environment_variables': get_environment_variables(),
        'network_info': get_network_info()
    }
    
    return report

def write_report_to_file(report, filename=None):
    """Write report to text file"""
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_report_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SYSTEM DEBUG REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {report['report_info']['generated_at']}\n")
        f.write(f"Purpose: {report['report_info']['purpose']}\n")
        f.write("=" * 80 + "\n\n")
        
        # System Information
        f.write("GENERAL SYSTEM INFORMATION\n")
        f.write("-" * 40 + "\n")
        for key, value in report['system_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Python Information
        f.write("PYTHON ENVIRONMENT INFORMATION\n")
        f.write("-" * 40 + "\n")
        for key, value in report['python_info'].items():
            if key == 'python_path':
                f.write(f"{key.replace('_', ' ').title()}:\n")
                for path in value:
                    f.write(f"  - {path}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Hardware Information
        f.write("HARDWARE INFORMATION\n")
        f.write("-" * 40 + "\n")
        for key, value in report['hardware_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # Installed Packages
        f.write("INSTALLED PACKAGES\n")
        f.write("-" * 40 + "\n")
        packages = report['installed_packages']
        if 'pip_packages' in packages:
            f.write(f"Total Packages: {packages.get('total_packages', 'Unknown')}\n\n")
            f.write("Package List:\n")
            for pkg, version in sorted(packages['pip_packages'].items()):
                f.write(f"  {pkg}: {version}\n")
        f.write("\n")
        
        # Environment Variables
        f.write("ENVIRONMENT VARIABLES\n")
        f.write("-" * 40 + "\n")
        for key, value in sorted(report['environment_variables'].items()):
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Network Information
        f.write("NETWORK INFORMATION\n")
        f.write("-" * 40 + "\n")
        for key, value in report['network_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    return filename

def create_debug_report(filename=None, include_json=True):
    """
    Create a comprehensive system debug report
    
    Args:
        filename (str, optional): Custom filename for the report. If None, generates timestamped filename
        include_json (bool): Whether to also create a JSON version of the report
    
    Returns:
        dict: Contains 'txt_file', 'json_file' (if created), and 'report_data'
    """
    try:
        # Generate report
        report = generate_debug_report()
        
        # Write to file
        txt_filename = write_report_to_file(report, filename)
        
        result = {
            'txt_file': txt_filename,
            'report_data': report,
            'success': True
        }
        
        # Also save as JSON for programmatic access
        if include_json:
            json_filename = txt_filename.replace('.txt', '.json')
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            result['json_file'] = json_filename
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'txt_file': None,
            'json_file': None,
            'report_data': None
        }
        