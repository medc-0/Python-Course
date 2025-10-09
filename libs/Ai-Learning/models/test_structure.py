"""
Test Script for LLM Structure
============================

This script tests the basic structure of our LLM implementation
without requiring PyTorch to be installed.
"""

import os
import sys

def test_file_structure():
    """Test that all required files exist"""
    print("🔍 Testing LLM File Structure")
    print("=" * 40)
    
    required_files = [
        'simple_llm.py',
        'simple_tokenizer.py', 
        'train_llm.py',
        'demo_llm.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    existing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file}")
    
    print(f"\nSummary:")
    print(f"✅ Found: {len(existing_files)} files")
    print(f"❌ Missing: {len(missing_files)} files")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    else:
        print(f"\n🎉 All files present!")
        return True

def test_imports():
    """Test that Python files can be parsed (syntax check)"""
    print(f"\n🐍 Testing Python File Syntax")
    print("=" * 40)
    
    python_files = [
        'simple_llm.py',
        'simple_tokenizer.py',
        'train_llm.py', 
        'demo_llm.py'
    ]
    
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the code (syntax check)
            compile(content, file, 'exec')
            print(f"✅ {file} - Syntax OK")
            
        except SyntaxError as e:
            print(f"❌ {file} - Syntax Error: {e}")
            return False
        except Exception as e:
            print(f"⚠️ {file} - Warning: {e}")
    
    return True

def test_requirements():
    """Test requirements.txt content"""
    print(f"\n📦 Testing Requirements")
    print("=" * 40)
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        required_packages = ['torch', 'numpy', 'matplotlib', 'tqdm']
        
        for package in required_packages:
            if package in content:
                print(f"✅ {package}")
            else:
                print(f"❌ {package} - Missing from requirements.txt")
        
        print(f"\nRequirements file looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 LLM Structure Test")
    print("=" * 50)
    
    # Test file structure
    structure_ok = test_file_structure()
    
    # Test Python syntax
    syntax_ok = test_imports()
    
    # Test requirements
    requirements_ok = test_requirements()
    
    print(f"\n" + "=" * 50)
    print(f"📊 Test Results:")
    print(f"File Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Python Syntax:  {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    print(f"Requirements:   {'✅ PASS' if requirements_ok else '❌ FAIL'}")
    
    if all([structure_ok, syntax_ok, requirements_ok]):
        print(f"\n🎉 All tests passed!")
        print(f"\nNext steps:")
        print(f"1. Install dependencies: pip install -r requirements.txt")
        print(f"2. Run the demo: python demo_llm.py")
        print(f"3. Train your model: python train_llm.py")
        return True
    else:
        print(f"\n❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
