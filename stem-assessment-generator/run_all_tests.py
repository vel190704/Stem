#!/usr/bin/env python3
"""
Comprehensive test runner for STEM Assessment Generator
Runs all test scripts and provides summary report
"""
import sys
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import settings

def run_command(command, description, timeout=60):
    """Run a command and capture output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        duration = time.time() - start_time
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {duration:.2f}s")
        
        if result.stdout:
            print(f"\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print(f"\nSTDERR:")
            print(result.stderr)
        
        return {
            'success': result.returncode == 0,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'description': description
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after {timeout}s")
        return {
            'success': False,
            'duration': timeout,
            'stdout': '',
            'stderr': f'Test timed out after {timeout}s',
            'description': description
        }
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return {
            'success': False,
            'duration': time.time() - start_time,
            'stdout': '',
            'stderr': str(e),
            'description': description
        }

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("="*60)
    print("PREREQUISITES CHECK")
    print("="*60)
    
    checks = []
    
    # Check Python version
    python_version = sys.version_info
    checks.append({
        'name': 'Python Version',
        'passed': python_version >= (3, 8),
        'details': f'{python_version.major}.{python_version.minor}.{python_version.micro}'
    })
    
    # Check required files exist
    required_files = [
        'config.py', 'models.py', 'pdf_processor.py', 
        'rag_pipeline.py', 'generator.py', 'app.py'
    ]
    
    for file in required_files:
        file_path = Path(file)
        checks.append({
            'name': f'File: {file}',
            'passed': file_path.exists(),
            'details': f'Size: {file_path.stat().st_size if file_path.exists() else 0} bytes'
        })
    
    # Check configuration
    checks.append({
        'name': 'Configuration',
        'passed': settings is not None,
        'details': f'Debug: {settings.DEBUG if settings else "N/A"}'
    })
    
    # Check OpenAI configuration
    checks.append({
        'name': 'OpenAI API Key',
        'passed': settings.is_openai_configured() if settings else False,
        'details': 'Configured' if (settings and settings.is_openai_configured()) else 'Not configured'
    })
    
    # Check directories
    if settings:
        for dir_name, dir_path in [('Upload', settings.upload_path), ('VectorDB', settings.vectordb_path)]:
            checks.append({
                'name': f'{dir_name} Directory',
                'passed': dir_path.exists() and dir_path.is_dir(),
                'details': str(dir_path)
            })
    
    # Print results
    for check in checks:
        status = "✓" if check['passed'] else "❌"
        print(f"  {status} {check['name']}: {check['details']}")
    
    failed_checks = [c for c in checks if not c['passed']]
    
    if failed_checks:
        print(f"\n❌ {len(failed_checks)} prerequisite(s) failed:")
        for check in failed_checks:
            print(f"     • {check['name']}")
        return False
    else:
        print(f"\n✓ All {len(checks)} prerequisites passed")
        return True

def run_all_tests():
    """Run all test scripts and collect results"""
    print("="*80)
    print("STEM ASSESSMENT GENERATOR - COMPREHENSIVE TEST SUITE")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    # Check prerequisites first
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues before running tests.")
        return False
    
    # Define test commands
    test_commands = [
        {
            'command': 'python test_pdf_processor.py',
            'description': 'PDF Processor Tests',
            'timeout': 120
        },
        {
            'command': 'python test_rag_pipeline.py',
            'description': 'RAG Pipeline Tests',
            'timeout': 180  # Longer timeout for OpenAI API calls
        },
        {
            'command': 'python -c "import app; print(\'App import successful\')"',
            'description': 'App Import Test',
            'timeout': 30
        },
        {
            'command': 'python -c "from generator import AssessmentGenerator; print(\'Generator import successful\')"',
            'description': 'Generator Import Test',
            'timeout': 30
        }
    ]
    
    # Run tests
    results = []
    total_duration = 0
    
    for test_config in test_commands:
        result = run_command(
            test_config['command'],
            test_config['description'],
            test_config['timeout']
        )
        results.append(result)
        total_duration += result['duration']
    
    # Generate summary report
    print("\n" + "="*80)
    print("TEST SUMMARY REPORT")
    print("="*80)
    
    passed_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Total tests run: {len(results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Total duration: {total_duration:.2f}s")
    
    if passed_tests:
        print(f"\n✓ PASSED TESTS:")
        for result in passed_tests:
            print(f"   • {result['description']} ({result['duration']:.2f}s)")
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for result in failed_tests:
            print(f"   • {result['description']} ({result['duration']:.2f}s)")
            if result['stderr']:
                print(f"     Error: {result['stderr'][:200]}...")
    
    # Overall status
    overall_success = len(failed_tests) == 0
    print(f"\n{'='*80}")
    if overall_success:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print("⚠️  SOME TESTS FAILED. Please review errors above.")
    print(f"{'='*80}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if not settings.is_openai_configured():
        print("   • Set OPENAI_API_KEY environment variable for full functionality")
        print("   • Get API key from: https://platform.openai.com/account/api-keys")
    
    if overall_success:
        print("   • Run 'python app.py' to start the web application")
        print("   • Access the interface at http://localhost:8000")
        print("   • Run 'python manual_testing_guide.py' for manual testing procedures")
    else:
        print("   • Fix failing tests before proceeding")
        print("   • Check error messages for specific issues")
        print("   • Ensure all dependencies are installed")
    
    return overall_success

def run_specific_test(test_name):
    """Run a specific test by name"""
    test_map = {
        'pdf': 'python test_pdf_processor.py',
        'rag': 'python test_rag_pipeline.py',
        'app': 'python -c "import app; print(\'App import successful\')"',
        'generator': 'python -c "from generator import AssessmentGenerator; print(\'Generator import successful\')"',
        'config': 'python -c "from config import settings; print(f\'Config loaded: {settings}\')"'
    }
    
    if test_name not in test_map:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {list(test_map.keys())}")
        return False
    
    result = run_command(
        test_map[test_name],
        f"{test_name.upper()} Test",
        120
    )
    
    return result['success']

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1].lower()
        success = run_specific_test(test_name)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = run_all_tests()
        
        # Print final instructions
        print(f"\nNext steps:")
        if success:
            print("1. python app.py                    # Start the application")
            print("2. Open http://localhost:8000       # Access web interface")
            print("3. Upload a PDF and test generation # End-to-end testing")
        else:
            print("1. Review error messages above")
            print("2. Fix any failing tests")
            print("3. Re-run: python run_all_tests.py")
        
        sys.exit(0 if success else 1)
