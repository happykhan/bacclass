#!/usr/bin/env python3
"""
Setup script to help configure Google Cloud BigQuery authentication
"""

import subprocess

def check_gcloud_installed():
    """Check if Google Cloud SDK is installed"""
    try:
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✓ Google Cloud SDK is installed")
        print(result.stdout.split('\n')[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Google Cloud SDK is not installed")
        print("\nInstall Google Cloud SDK:")
        print("  macOS: brew install google-cloud-sdk")
        print("  Or visit: https://cloud.google.com/sdk/docs/install")
        return False

def check_authentication():
    """Check if user is authenticated"""
    try:
        result = subprocess.run(['gcloud', 'auth', 'list'], 
                              capture_output=True, text=True, check=True)
        if "No credentialed accounts" in result.stdout:
            print("✗ No authenticated accounts found")
            return False
        else:
            print("✓ Authentication configured")
            return True
    except subprocess.CalledProcessError:
        print("✗ Error checking authentication")
        return False

def check_project():
    """Check if a project is configured"""
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                              capture_output=True, text=True, check=True)
        project = result.stdout.strip()
        if project and project != "(unset)":
            print(f"✓ Project configured: {project}")
            return True
        else:
            print("✗ No project configured")
            return False
    except subprocess.CalledProcessError:
        print("✗ Error checking project configuration")
        return False

def setup_authentication():
    """Guide user through authentication setup"""
    print("\nSetting up Google Cloud authentication...")
    print("\n1. Authenticate with Google Cloud:")
    print("   gcloud auth application-default login")
    
    choice = input("\nRun authentication now? (y/n): ").lower()
    if choice == 'y':
        try:
            subprocess.run(['gcloud', 'auth', 'application-default', 'login'], check=True)
            print("✓ Authentication completed")
        except subprocess.CalledProcessError:
            print("✗ Authentication failed")

def setup_project():
    """Guide user through project setup"""
    print("\n2. Set up a Google Cloud project:")
    print("   Option A: Use existing project")
    print("   Option B: Create new project")
    
    project_id = input("\nEnter your Google Cloud project ID (or press Enter to skip): ").strip()
    if project_id:
        try:
            subprocess.run(['gcloud', 'config', 'set', 'project', project_id], check=True)
            print(f"✓ Project set to: {project_id}")
        except subprocess.CalledProcessError:
            print("✗ Failed to set project")

def enable_bigquery():
    """Enable BigQuery API"""
    print("\n3. Enable BigQuery API:")
    choice = input("Enable BigQuery API for your project? (y/n): ").lower()
    if choice == 'y':
        try:
            subprocess.run(['gcloud', 'services', 'enable', 'bigquery.googleapis.com'], check=True)
            print("✓ BigQuery API enabled")
        except subprocess.CalledProcessError:
            print("✗ Failed to enable BigQuery API")

def test_bigquery_access():
    """Test BigQuery access with a simple query"""
    print("\n4. Testing BigQuery access...")
    
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        
        # Simple test query
        query = """
        SELECT 1 as test_value
        LIMIT 1
        """
        
        query_job = client.query(query)
        results = list(query_job)
        
        if results:
            print("✓ BigQuery access working!")
            return True
        else:
            print("✗ BigQuery test failed")
            return False
            
    except Exception as e:
        print(f"✗ BigQuery test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Google Cloud BigQuery Setup Helper")
    print("=" * 40)
    
    # Check prerequisites
    if not check_gcloud_installed():
        return
    
    # Check current status
    auth_ok = check_authentication()
    project_ok = check_project()
    
    if auth_ok and project_ok:
        print("\n✓ Google Cloud seems to be configured")
        test_bigquery_access()
        return
    
    # Setup if needed
    if not auth_ok:
        setup_authentication()
    
    if not project_ok:
        setup_project()
    
    enable_bigquery()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Test your BigQuery script:")
    print("   python bacclass/bigquery.py --accession SAMN49579103")
    print("2. Fetch Salmonella data:")
    print("   python bacclass/bigquery.py --limit 5 --output salmonella_data.csv")

if __name__ == "__main__":
    main()
