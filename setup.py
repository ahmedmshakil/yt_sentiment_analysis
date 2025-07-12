#!/usr/bin/env python3
"""
Setup script for the RAG system.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 RAG System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("\n🔧 Setting up environment file...")
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as example_file:
                example_content = example_file.read()
            
            with open('.env', 'w') as env_file:
                env_file.write(example_content)
            
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file and add your Gemini API key")
        else:
            print("❌ .env.example file not found")
            sys.exit(1)
    else:
        print("✅ .env file already exists")
    
    # Run tests
    print("\n🧪 Running tests...")
    if run_command("python test_rag.py", "Running system tests"):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your Gemini API key")
        print("2. Run 'python main.py' for command-line interface")
        print("3. Run 'streamlit run streamlit_app.py' for web interface")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Please check the test output above for issues")

if __name__ == "__main__":
    main()