"""
System validation and health checks for STEM Assessment Generator
Validates configuration, dependencies, and system requirements
"""
import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging

from config import settings
from exceptions import (
    ConfigurationError, DependencyError, OpenAIKeyError, 
    FileSystemError, VectorDatabaseError
)

logger = logging.getLogger(__name__)

class SystemValidator:
    """Comprehensive system validation and health checking"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks"""
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "checks": {}
        }
        
        # Core validation checks
        checks = [
            ("python_version", self._check_python_version),
            ("dependencies", self._check_dependencies),
            ("openai_config", self._check_openai_configuration),
            ("file_permissions", self._check_file_permissions),
            ("disk_space", self._check_disk_space),
            ("vector_store", self._check_vector_store),
            ("environment", self._check_environment_variables),
            ("optional_features", self._check_optional_features)
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                self.validation_results["checks"][check_name] = {
                    "status": "passed" if result["success"] else "failed",
                    "details": result
                }
            except Exception as e:
                self.validation_results["checks"][check_name] = {
                    "status": "error",
                    "details": {"error": str(e)}
                }
                self.errors.append(f"{check_name}: {str(e)}")
        
        # Overall system health
        failed_checks = [name for name, result in self.validation_results["checks"].items() 
                        if result["status"] in ["failed", "error"]]
        
        self.validation_results["overall_status"] = "healthy" if not failed_checks else "degraded"
        self.validation_results["failed_checks"] = failed_checks
        self.validation_results["errors"] = self.errors
        self.validation_results["warnings"] = self.warnings
        
        return self.validation_results
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility"""
        required_major, required_minor = 3, 8
        current_major, current_minor = sys.version_info[:2]
        
        is_compatible = (current_major > required_major or 
                        (current_major == required_major and current_minor >= required_minor))
        
        return {
            "success": is_compatible,
            "current_version": f"{current_major}.{current_minor}",
            "required_version": f"{required_major}.{required_minor}+",
            "compatible": is_compatible
        }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check if all required dependencies are installed"""
        required_packages = {
            "fastapi": ("fastapi", "0.68.0"),
            "uvicorn": ("uvicorn", "0.15.0"),
            "pydantic": ("pydantic", "1.8.0"),
            "openai": ("openai", "1.0.0"),
            "chromadb": ("chromadb", "0.4.0"),
            "PyPDF2": ("PyPDF2", "3.0.0"),
            "numpy": ("numpy", "1.21.0"),
            "python-multipart": ("multipart", "0.0.5")
        }
        
        optional_packages = {
            "reportlab": ("reportlab", "4.0.0"),
            "python-docx": ("docx", "1.1.0"),
            "textstat": ("textstat", "0.7.0"),
            "scikit-learn": ("sklearn", "1.3.0"),
            "redis": ("redis", "5.0.0")
        }
        
        installed = {}
        missing_required = []
        missing_optional = []
        
        # Check required packages
        for package_name, (import_name, min_version) in required_packages.items():
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, "__version__", "unknown")
                installed[package_name] = version
            except ImportError:
                missing_required.append(package_name)
        
        # Check optional packages
        for package_name, (import_name, min_version) in optional_packages.items():
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, "__version__", "unknown")
                installed[package_name] = version
            except ImportError:
                missing_optional.append(package_name)
        
        if missing_optional:
            self.warnings.append(f"Optional packages missing: {', '.join(missing_optional)}")
        
        return {
            "success": len(missing_required) == 0,
            "installed_packages": installed,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "total_required": len(required_packages),
            "total_installed": len(installed)
        }
    
    def _check_openai_configuration(self) -> Dict[str, Any]:
        """Check OpenAI API configuration"""
        try:
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                return {
                    "success": False,
                    "configured": False,
                    "error": "OPENAI_API_KEY not set in environment"
                }
            
            # Basic key format validation
            if not api_key.startswith(('sk-', 'sk-proj-')):
                self.warnings.append("OpenAI API key format may be incorrect")
            
            # Try to make a test call (without actually calling API to save quota)
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            return {
                "success": True,
                "configured": True,
                "key_format": "valid" if api_key.startswith(('sk-', 'sk-proj-')) else "unknown",
                "model": settings.OPENAI_MODEL,
                "embedding_model": settings.OPENAI_EMBEDDING_MODEL
            }
            
        except Exception as e:
            return {
                "success": False,
                "configured": False,
                "error": str(e)
            }
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file system permissions"""
        directories_to_check = [
            settings.upload_path,
            settings.vectordb_path,
            Path.cwd()  # Current working directory
        ]
        
        permission_results = {}
        all_writable = True
        
        for directory in directories_to_check:
            try:
                # Ensure directory exists
                directory.mkdir(parents=True, exist_ok=True)
                
                # Test write permission
                test_file = directory / f".write_test_{datetime.now().timestamp()}"
                test_file.write_text("test")
                test_file.unlink()
                
                permission_results[str(directory)] = {
                    "exists": True,
                    "writable": True,
                    "readable": True
                }
                
            except PermissionError:
                permission_results[str(directory)] = {
                    "exists": directory.exists(),
                    "writable": False,
                    "readable": directory.exists()
                }
                all_writable = False
                
            except Exception as e:
                permission_results[str(directory)] = {
                    "exists": False,
                    "writable": False,
                    "readable": False,
                    "error": str(e)
                }
                all_writable = False
        
        return {
            "success": all_writable,
            "directories": permission_results,
            "total_checked": len(directories_to_check)
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import shutil
            
            # Check space for upload directory
            upload_space = shutil.disk_usage(settings.upload_path)
            vectordb_space = shutil.disk_usage(settings.vectordb_path)
            
            # Convert to GB
            upload_free_gb = upload_space.free / (1024**3)
            vectordb_free_gb = vectordb_space.free / (1024**3)
            
            # Minimum required space (1GB)
            min_required_gb = 1.0
            
            sufficient_space = (upload_free_gb >= min_required_gb and 
                              vectordb_free_gb >= min_required_gb)
            
            if not sufficient_space:
                self.warnings.append(f"Low disk space detected")
            
            return {
                "success": sufficient_space,
                "upload_dir_free_gb": round(upload_free_gb, 2),
                "vectordb_dir_free_gb": round(vectordb_free_gb, 2),
                "min_required_gb": min_required_gb,
                "sufficient": sufficient_space
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_vector_store(self) -> Dict[str, Any]:
        """Check vector store connectivity"""
        try:
            from rag_pipeline import RAGPipeline
            
            # Try to initialize the vector store
            rag = RAGPipeline()
            collections = rag.list_collections()
            
            return {
                "success": True,
                "connected": True,
                "collections_count": len(collections),
                "collections": collections[:5]  # Show first 5 collections
            }
            
        except Exception as e:
            return {
                "success": False,
                "connected": False,
                "error": str(e)
            }
    
    def _check_environment_variables(self) -> Dict[str, Any]:
        """Check important environment variables"""
        important_vars = [
            "OPENAI_API_KEY",
            "PYTHONPATH",
            "PATH"
        ]
        
        optional_vars = [
            "REDIS_URL",
            "DATABASE_URL",
            "LOG_LEVEL"
        ]
        
        env_status = {}
        missing_required = []
        
        for var in important_vars:
            value = os.getenv(var)
            env_status[var] = {
                "set": bool(value),
                "length": len(value) if value else 0
            }
            if not value and var == "OPENAI_API_KEY":
                missing_required.append(var)
        
        for var in optional_vars:
            value = os.getenv(var)
            env_status[var] = {
                "set": bool(value),
                "optional": True
            }
        
        return {
            "success": len(missing_required) == 0,
            "environment_variables": env_status,
            "missing_required": missing_required
        }
    
    def _check_optional_features(self) -> Dict[str, Any]:
        """Check availability of optional features"""
        features = {}
        
        # PDF export capability
        try:
            import reportlab
            features["pdf_export"] = {"available": True, "version": reportlab.Version}
        except ImportError:
            features["pdf_export"] = {"available": False, "reason": "reportlab not installed"}
        
        # Word export capability
        try:
            import docx
            features["docx_export"] = {"available": True}
        except ImportError:
            features["docx_export"] = {"available": False, "reason": "python-docx not installed"}
        
        # Analytics capability
        try:
            import textstat
            import sklearn
            features["analytics"] = {"available": True}
        except ImportError:
            features["analytics"] = {"available": False, "reason": "textstat or scikit-learn not installed"}
        
        # Redis caching
        try:
            import redis
            features["redis_cache"] = {"available": True}
        except ImportError:
            features["redis_cache"] = {"available": False, "reason": "redis not installed"}
        
        available_count = sum(1 for f in features.values() if f["available"])
        
        return {
            "success": True,  # Optional features don't affect success
            "features": features,
            "available_features": available_count,
            "total_features": len(features)
        }
    
    def get_startup_requirements(self) -> Tuple[List[str], List[str]]:
        """Get list of critical errors and warnings for startup"""
        critical_errors = []
        startup_warnings = []
        
        validation = self.validate_all()
        
        # Check for critical issues that prevent startup
        failed_checks = validation.get("failed_checks", [])
        
        if "dependencies" in failed_checks:
            critical_errors.append("Required dependencies are missing")
        
        if "file_permissions" in failed_checks:
            critical_errors.append("Cannot write to required directories")
        
        if "openai_config" in failed_checks:
            startup_warnings.append("OpenAI API not configured - features will be limited")
        
        if validation.get("warnings"):
            startup_warnings.extend(validation["warnings"])
        
        return critical_errors, startup_warnings
    
    def generate_health_report(self) -> str:
        """Generate a human-readable health report"""
        validation = self.validate_all()
        
        report = [
            "=== STEM Assessment Generator Health Report ===",
            f"Generated: {validation['timestamp']}",
            f"Overall Status: {validation['overall_status'].upper()}",
            ""
        ]
        
        # System info
        report.append("System Information:")
        report.append(f"  Python Version: {validation['python_version']}")
        report.append("")
        
        # Check results
        for check_name, result in validation["checks"].items():
            status_icon = "✅" if result["status"] == "passed" else "❌"
            report.append(f"{status_icon} {check_name.replace('_', ' ').title()}: {result['status']}")
        
        # Errors and warnings
        if validation["errors"]:
            report.append("\nErrors:")
            for error in validation["errors"]:
                report.append(f"  ❌ {error}")
        
        if validation["warnings"]:
            report.append("\nWarnings:")
            for warning in validation["warnings"]:
                report.append(f"  ⚠️  {warning}")
        
        return "\n".join(report)

# Global validator instance
system_validator = SystemValidator()

def validate_startup() -> bool:
    """Quick startup validation - returns True if system can start"""
    critical_errors, warnings = system_validator.get_startup_requirements()
    
    if critical_errors:
        logger.error("Critical errors prevent startup:")
        for error in critical_errors:
            logger.error(f"  - {error}")
        return False
    
    if warnings:
        logger.warning("Startup warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    return True

def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    return system_validator.validate_all()
