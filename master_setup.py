#!/usr/bin/env python3
"""
MASTER SETUP SCRIPT
Run this single script to set up everything
Usage: python master_setup.py
"""

import subprocess
import sys
from pathlib import Path
import time

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}→ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def run_command(cmd, description, ignore_errors=False):
    """Run shell command with nice output"""
    print_info(f"{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print_success(f"{description} - Complete")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print_warning(f"{description} - Skipped (non-critical)")
            return True
        else:
            print_error(f"{description} - Failed")
            print(f"  Error: {e.stderr}")
            return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error("Python 3.8+ required!")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    print_success(f"Python version: {version.major}.{version.minor}")
    return True

def setup_project():
    """Complete project setup"""
    
    print_header("SMART RETAIL CV - MASTER SETUP")
    
    start_time = time.time()
    
    # Step 1: Check Python version
    print_header("Step 1: Checking Requirements")
    if not check_python_version():
        return False
    
    # Step 2: Create virtual environment
    print_header("Step 2: Creating Virtual Environment")
    if Path("venv").exists():
        print_warning("Virtual environment already exists, skipping...")
    else:
        run_command("python -m venv venv", "Creating virtual environment")
    
    # Determine activation command based on OS
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    print_info(f"To activate: {activate_cmd}")
    
    # Step 3: Install dependencies
    print_header("Step 3: Installing Dependencies")
    
    print_info("Installing PyTorch (CPU version for speed)...")
    run_command(
        f"{pip_cmd} install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "Installing PyTorch"
    )
    
    print_info("Installing other requirements...")
    run_command(
        f"{pip_cmd} install ultralytics opencv-python opencv-contrib-python",
        "Installing CV packages"
    )
    run_command(
        f"{pip_cmd} install fastapi uvicorn streamlit",
        "Installing web frameworks"
    )
    run_command(
        f"{pip_cmd} install pandas numpy matplotlib seaborn plotly",
        "Installing data science packages"
    )
    run_command(
        f"{pip_cmd} install albumentations pillow tqdm pyyaml scipy filterpy scikit-image",
        "Installing utilities"
    )
    run_command(
        f"{pip_cmd} install python-multipart python-dotenv requests",
        "Installing API dependencies"
    )
    
    # Step 4: Create project structure
    print_header("Step 4: Creating Project Structure")
    run_command(f"{python_cmd} scripts/setup_project.py", "Setting up directories")
    
    # Step 5: Generate synthetic data
    print_header("Step 5: Generating Synthetic Videos")
    print_warning("This will take 2-3 minutes...")
    run_command(
        f"{python_cmd} scripts/generate_synthetic_video.py",
        "Generating sample videos"
    )
    
    # Step 6: Prepare data
    print_header("Step 6: Preparing Training Data")
    run_command(
        f"{python_cmd} src/data_preparation/prepare_data.py",
        "Processing videos and creating dataset"
    )
    
    # Step 7: Quick training (optional - can be run separately)
    print_header("Step 7: Training Models (Optional)")
    print_info("You can train models now or later manually")
    response = input("Train models now? This will take 15-20 minutes (y/n): ").lower()
    
    if response == 'y':
        print_info("Starting model training...")
        run_command(
            f"{python_cmd} src/training/train_detector.py --epochs 10 --batch 16",
            "Training person detector"
        )
        run_command(
            f"{python_cmd} src/training/train_inventory.py --epochs 5 --batch 32",
            "Training inventory classifier"
        )
        print_success("Models trained successfully!")
    else:
        print_warning("Skipping model training. Run manually:")
        print(f"  {python_cmd} src/training/train_detector.py --epochs 10")
        print(f"  {python_cmd} src/training/train_inventory.py --epochs 5")
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print_header("SETUP COMPLETE!")
    
    print_success(f"Total setup time: {elapsed_time/60:.1f} minutes")
    print("\n" + "="*60)
    print(f"{Colors.BOLD}NEXT STEPS:{Colors.ENDC}")
    print("="*60)
    
    print(f"\n1. Activate virtual environment:")
    print(f"   {Colors.OKCYAN}{activate_cmd}{Colors.ENDC}")
    
    if response != 'y':
        print(f"\n2. Train models:")
        print(f"   {Colors.OKCYAN}{python_cmd} src/training/train_detector.py --epochs 10{Colors.ENDC}")
        print(f"   {Colors.OKCYAN}{python_cmd} src/training/train_inventory.py --epochs 5{Colors.ENDC}")
    
    print(f"\n3. Run inference:")
    print(f"   {Colors.OKCYAN}{python_cmd} src/inference/run_inference.py{Colors.ENDC}")
    
    print(f"\n4. Start API server (Terminal 1):")
    print(f"   {Colors.OKCYAN}{python_cmd} src/api/main.py{Colors.ENDC}")
    
    print(f"\n5. Start Dashboard (Terminal 2):")
    print(f"   {Colors.OKCYAN}streamlit run dashboard/app.py{Colors.ENDC}")
    
    print(f"\n6. Access services:")
    print(f"   API Docs: {Colors.OKCYAN}http://localhost:8000/docs{Colors.ENDC}")
    print(f"   Dashboard: {Colors.OKCYAN}http://localhost:8501{Colors.ENDC}")
    
    print("\n" + "="*60)
    print(f"{Colors.OKGREEN}{Colors.BOLD}🎉 Ready to build your resume project!{Colors.ENDC}")
    print("="*60 + "\n")
    
    return True

def main():
    try:
        setup_project()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
