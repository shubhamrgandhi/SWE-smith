#!/usr/bin/env python3
"""
Unify SWE-smith datasets with trajectory data and image validation.

- Merges SWE-smith-trajectories with SWE-smith based on common instance_ids
- Renames 'messages' field to 'expert_messages' for trajectory data
- Validates Docker image availability and updates image_name fields
- Preserves all duplicate rows and handles cross-product merging
- Supports local saving and HuggingFace Hub upload
"""

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import argparse
import requests
import json
import os
from typing import Set, Dict, Any

def load_datasets():
    """Load both SWE-bench datasets."""
    print("Loading SWE-bench/SWE-smith-trajectories...")
    trajectories_dataset = load_dataset("SWE-bench/SWE-smith-trajectories")
    
    print("Loading SWE-bench/SWE-smith...")
    smith_dataset = load_dataset("SWE-bench/SWE-smith")
    
    return trajectories_dataset, smith_dataset

def get_instance_ids(dataset) -> Set[str]:
    """Extract all instance_ids from a dataset."""
    instance_ids = set()
    
    # Handle both single split and multi-split datasets
    if isinstance(dataset, DatasetDict):
        for split_name, split_data in dataset.items():
            instance_ids.update(split_data['instance_id'])
    else:
        instance_ids.update(dataset['instance_id'])
    
    return instance_ids

def analyze_overlap(trajectories_dataset, smith_dataset):
    """Analyze the overlap between the two datasets."""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Convert datasets to pandas for easier analysis
    if isinstance(trajectories_dataset, DatasetDict):
        trajectories_df = trajectories_dataset[list(trajectories_dataset.keys())[0]].to_pandas()
    else:
        trajectories_df = trajectories_dataset.to_pandas()
    
    if isinstance(smith_dataset, DatasetDict):
        smith_df = smith_dataset[list(smith_dataset.keys())[0]].to_pandas()
    else:
        smith_df = smith_dataset.to_pandas()
    
    # Get instance_ids from both datasets (unique sets)
    trajectories_ids = set(trajectories_df['instance_id'].unique())
    smith_ids = set(smith_df['instance_id'].unique())
    
    # Get total row counts
    trajectories_total_rows = len(trajectories_df)
    smith_total_rows = len(smith_df)
    
    print(f"SWE-smith-trajectories:")
    print(f"  - Total rows: {trajectories_total_rows:,}")
    print(f"  - Unique instance_ids: {len(trajectories_ids):,}")
    
    print(f"SWE-smith:")
    print(f"  - Total rows: {smith_total_rows:,}")
    print(f"  - Unique instance_ids: {len(smith_ids):,}")
    
    # Find intersection and missing IDs
    common_ids = trajectories_ids.intersection(smith_ids)
    missing_ids = trajectories_ids - smith_ids
    
    print(f"Common unique instance_ids: {len(common_ids):,}")
    print(f"Missing from SWE-smith: {len(missing_ids):,}")
    
    # Print the missing IDs for verification
    if missing_ids:
        print(f"\nMissing instance_ids from SWE-smith:")
        total_missing_rows = 0
        for i, missing_id in enumerate(sorted(missing_ids), 1):
            # Count how many rows this missing_id has in trajectories
            row_count = len(trajectories_df[trajectories_df['instance_id'] == missing_id])
            total_missing_rows += row_count
            print(f"  {i}. {missing_id} (appears {row_count} times)")
        
        print(f"\nTotal missing rows: {total_missing_rows}")
        
        # Double-check by manually searching for each missing ID
        print(f"\nDouble-checking missing IDs in SWE-smith dataset:")
        for missing_id in sorted(missing_ids):
            # Check if it exists in any form (case-insensitive, whitespace, etc.)
            exact_match = missing_id in smith_df['instance_id'].values
            case_insensitive_match = missing_id.lower() in smith_df['instance_id'].str.lower().values
            stripped_match = missing_id.strip() in smith_df['instance_id'].str.strip().values
            
            print(f"  {missing_id}:")
            print(f"    - Exact match: {exact_match}")
            print(f"    - Case-insensitive match: {case_insensitive_match}")
            print(f"    - Stripped match: {stripped_match}")
            
            # Look for partial matches (in case there are slight variations)
            partial_matches = smith_df[smith_df['instance_id'].str.contains(missing_id.split('__')[0] if '__' in missing_id else missing_id[:20], case=False, na=False)]
            if len(partial_matches) > 0:
                print(f"    - Partial matches found: {len(partial_matches)}")
                print(f"    - Partial match examples: {partial_matches['instance_id'].head(3).tolist()}")
            else:
                print(f"    - No partial matches found")
    
    # Calculate how many rows will be in the final dataset
    trajectories_matching_rows = len(trajectories_df[trajectories_df['instance_id'].isin(common_ids)])
    smith_matching_rows = len(smith_df[smith_df['instance_id'].isin(common_ids)])
    
    print(f"\nRows with matching instance_ids:")
    print(f"  - From trajectories: {trajectories_matching_rows:,}")
    print(f"  - From smith: {smith_matching_rows:,}")
    
    # Calculate percentages
    trajectories_coverage = len(common_ids) / len(trajectories_ids) * 100
    smith_coverage = len(common_ids) / len(smith_ids) * 100
    
    print(f"\nCoverage of unique instance_ids:")
    print(f"  - Trajectories dataset: {trajectories_coverage:.1f}%")
    print(f"  - Smith dataset: {smith_coverage:.1f}%")
    
    return common_ids, trajectories_df, smith_df

def create_unified_dataset(trajectories_df, smith_df, common_ids: Set[str]):
    """Create the unified dataset with matching instance_ids."""
    print("\n" + "="*60)
    print("CREATING UNIFIED DATASET")
    print("="*60)
    
    # Filter both datasets to only include common instance_ids
    # This retains ALL rows with matching instance_ids (including duplicates)
    trajectories_filtered = trajectories_df[trajectories_df['instance_id'].isin(common_ids)].copy()
    smith_filtered = smith_df[smith_df['instance_id'].isin(common_ids)].copy()
    
    print(f"Filtered trajectories rows: {len(trajectories_filtered):,}")
    print(f"Filtered smith rows: {len(smith_filtered):,}")
    
    # Check for duplicates in each dataset
    trajectories_duplicates = trajectories_filtered['instance_id'].duplicated().sum()
    smith_duplicates = smith_filtered['instance_id'].duplicated().sum()
    
    print(f"Duplicate instance_ids in trajectories: {trajectories_duplicates:,}")
    print(f"Duplicate instance_ids in smith: {smith_duplicates:,}")
    
    # Rename 'messages' field to 'expert_messages' in trajectories data
    if 'messages' in trajectories_filtered.columns:
        trajectories_filtered = trajectories_filtered.rename(columns={'messages': 'expert_messages'})
        print("Renamed 'messages' field to 'expert_messages'")
    
    # For the merge, we need to handle the case where there might be multiple rows
    # with the same instance_id in both datasets. We'll do a cross-join for each instance_id
    # to ensure we get all combinations
    
    unified_rows = []
    
    for instance_id in common_ids:
        # Get all rows for this instance_id from both datasets
        traj_rows = trajectories_filtered[trajectories_filtered['instance_id'] == instance_id]
        smith_rows = smith_filtered[smith_filtered['instance_id'] == instance_id]
        
        # Create all combinations (cross product)
        for _, smith_row in smith_rows.iterrows():
            for _, traj_row in traj_rows.iterrows():
                # Start with the smith row (all fields)
                unified_row = smith_row.to_dict()
                
                # Add the expert_messages from trajectories
                if 'expert_messages' in traj_row:
                    unified_row['expert_messages'] = traj_row['expert_messages']
                
                # Add any other fields from trajectories that aren't in smith
                for col in traj_row.index:
                    if col not in smith_row.index and col != 'instance_id':
                        unified_row[col] = traj_row[col]
                
                unified_rows.append(unified_row)
    
    # Convert to DataFrame
    unified_df = pd.DataFrame(unified_rows)
    
    print(f"Final unified dataset rows: {len(unified_df):,}")
    print(f"Columns in unified dataset: {len(unified_df.columns)}")
    print(f"Column names: {list(unified_df.columns)}")
    
    # Show some statistics about the final dataset
    final_unique_ids = unified_df['instance_id'].nunique()
    final_duplicate_count = len(unified_df) - final_unique_ids
    
    print(f"Final dataset statistics:")
    print(f"  - Unique instance_ids: {final_unique_ids:,}")
    print(f"  - Total rows: {len(unified_df):,}")
    print(f"  - Duplicate rows (same instance_id): {final_duplicate_count:,}")
    
    # Convert back to HuggingFace Dataset
    unified_dataset = Dataset.from_pandas(unified_df)
    
    return unified_dataset

def get_docker_hub_login():
    """Get Docker Hub credentials from config file"""
    docker_config_path = os.path.expanduser("~/.docker/config.json")
    try:
        with open(docker_config_path, "r") as config_file:
            docker_config = json.load(config_file)
        auths = docker_config.get("auths", {})
        docker_hub = auths.get("https://index.docker.io/v1/")
        if not docker_hub:
            print("Warning: Docker Hub credentials not found. Will check images without authentication.")
            return None, None
        # The token is encoded in Base64 (username:password), decode it
        from base64 import b64decode
        auth_token = docker_hub.get("auth")
        if not auth_token:
            print("Warning: No auth token found in Docker config.")
            return None, None
        decoded_auth = b64decode(auth_token).decode("utf-8")
        username, password = decoded_auth.split(":", 1)
        return username, password
    except FileNotFoundError:
        print("Warning: Docker config file not found. Will check images without authentication.")
        return None, None
    except Exception as e:
        print(f"Warning: Error retrieving Docker Hub credentials: {e}")
        return None, None

def get_dockerhub_token(username, password):
    """Get DockerHub authentication token"""
    if not username or not password:
        return None
    try:
        auth_url = "https://hub.docker.com/v2/users/login"
        auth_data = {"username": username, "password": password}
        response = requests.post(auth_url, json=auth_data)
        response.raise_for_status()
        return response.json()["token"]
    except Exception as e:
        print(f"Warning: Could not get Docker Hub token: {e}")
        return None

def check_docker_image_exists(org, repo_name, tag="latest", token=None):
    """Check if a Docker image exists on Docker Hub"""
    try:
        # Try Docker Hub API first
        url = f"https://hub.docker.com/v2/repositories/{org}/{repo_name}/tags/{tag}/"
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            # Fallback to Docker Registry API
            registry_url = f"https://registry.hub.docker.com/v2/{org}/{repo_name}/manifests/{tag}"
            registry_response = requests.head(registry_url)
            return registry_response.status_code == 200
    except Exception as e:
        print(f"Warning: Could not check image {org}/{repo_name}:{tag} - {e}")
        return False

def transform_image_name(image_name):
    """Apply the same transformation logic as in the Docker download script"""
    # The images appear to be stored with the full name including swesmith.x86_64. prefix
    # Based on HuggingFace discussions, the format is: jyangballin/swesmith.x86_64.{repo}.{hash}
    
    transformations = []
    
    # 1. Try as-is (keep the original format)
    if image_name.startswith("swesmith.x86_64."):
        transformations.append(image_name)
    else:
        # If it doesn't have the prefix, add it
        transformations.append(f"swesmith.x86_64.{image_name}")
    
    # 2. Try without the swesmith.x86_64. prefix (our original logic)
    cleaned_name = image_name
    if cleaned_name.startswith("swesmith.x86_64."):
        cleaned_name = cleaned_name.replace("swesmith.x86_64.", "")
    elif cleaned_name.startswith("swesmith."):
        cleaned_name = cleaned_name.replace("swesmith.", "")
    elif cleaned_name.startswith("swesmith_"):
        cleaned_name = cleaned_name.replace("swesmith_", "")
    
    # Apply the Docker download script transformations to the cleaned name
    # These match the logic from the original download script
    possible_cleaned_names = [
        cleaned_name.replace("__", "_1776_"),  # Replace __ with _1776_
        cleaned_name,  # Keep original
        cleaned_name.replace("/", "_1776_"),  # Replace / with _1776_
    ]
    
    # Add all variations with different prefixes
    for clean_name in possible_cleaned_names:
        # With swesmith_ prefix
        transformations.append(f"swesmith_{clean_name}")
        # With full swesmith.x86_64. prefix  
        transformations.append(f"swesmith.x86_64.{clean_name}")
    
    # 3. Try reverse transformation: if it has _1776_, try replacing back to __
    if "_1776_" in cleaned_name:
        reverse_name = cleaned_name.replace("_1776_", "__")
        transformations.append(f"swesmith_{reverse_name}")
        transformations.append(f"swesmith.x86_64.{reverse_name}")
    
    # 4. Try without any version suffix (remove everything after last dot if it looks like a version)
    if "." in cleaned_name:
        parts = cleaned_name.split(".")
        if len(parts) >= 2 and len(parts[-1]) >= 6:  # Looks like a hash/version
            base_name = ".".join(parts[:-1])
            
            # Apply same transformations to base name
            base_variants = [
                base_name.replace("__", "_1776_"),
                base_name,
                base_name.replace("/", "_1776_"),
            ]
            
            for base_variant in base_variants:
                transformations.append(f"swesmith_{base_variant}")
                transformations.append(f"swesmith.x86_64.{base_variant}")
                
            # Try reverse on base name too
            if "_1776_" in base_name:
                reverse_base = base_name.replace("_1776_", "__")
                transformations.append(f"swesmith_{reverse_base}")
                transformations.append(f"swesmith.x86_64.{reverse_base}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_transformations = []
    for trans in transformations:
        if trans not in seen:
            seen.add(trans)
            unique_transformations.append(trans)
    
    return unique_transformations

def check_and_update_image_names(unified_df, docker_org="jyangballin", tag="latest"):
    """Check Docker image existence and update image_name field"""
    print("\n" + "="*60)
    print("CHECKING DOCKER IMAGES")
    print("="*60)
    
    # Get Docker Hub credentials
    username, password = get_docker_hub_login()
    token = get_dockerhub_token(username, password)
    
    if 'image_name' not in unified_df.columns:
        print("Warning: 'image_name' field not found in dataset")
        return unified_df
    
    # Get unique image names
    unique_images = unified_df['image_name'].unique()
    print(f"Found {len(unique_images)} unique image names to check")
    
    # Track results
    image_status = {}
    successful_transformations = {}
    
    for i, image_name in enumerate(unique_images, 1):
        print(f"Checking {i}/{len(unique_images)}: {image_name}")
        
        # Generate possible transformations
        possible_names = transform_image_name(image_name)
        
        found = False
        working_name = None
        
        for candidate_name in possible_names:
            exists = check_docker_image_exists(docker_org, candidate_name, tag, token)
            
            if exists:
                print(f"  ✓ Found: {docker_org}/{candidate_name}:{tag}")
                found = True
                working_name = f"{docker_org}/{candidate_name}"
                break
            else:
                print(f"  ✗ Not found: {docker_org}/{candidate_name}:{tag}")
        
        image_status[image_name] = found
        if found:
            successful_transformations[image_name] = working_name
    
    # Update the dataset
    print(f"\nUpdating image_name field...")
    
    # Save original image names
    unified_df['image_name_original'] = unified_df['image_name'].copy()
    
    # Update image_name field with working transformations
    updated_count = 0
    for original_name, working_name in successful_transformations.items():
        mask = unified_df['image_name_original'] == original_name
        count = mask.sum()
        if count > 0:
            unified_df.loc[mask, 'image_name'] = working_name
            updated_count += count
            print(f"  Updated {count} rows: '{original_name}' -> '{working_name}'")
    
    # Add image_exists flag
    unified_df['image_exists'] = unified_df['image_name_original'].map(image_status)
    
    print(f"Total rows updated: {updated_count}")
    print(f"Total rows with existing images: {unified_df['image_exists'].sum()}")
    print(f"Total rows with missing images: {(~unified_df['image_exists']).sum()}")
    
    # Print summary
    total_images = len(unique_images)
    existing_images = sum(image_status.values())
    missing_images = total_images - existing_images
    
    print(f"\nDocker Image Summary:")
    print(f"  - Total unique images: {total_images}")
    print(f"  - Images found: {existing_images}")
    print(f"  - Images missing: {missing_images}")
    print(f"  - Success rate: {existing_images/total_images*100:.1f}%")
    
    if missing_images > 0:
        print(f"\nMissing images:")
        missing_image_names = [img for img, exists in image_status.items() if not exists]
        for img in missing_image_names[:10]:  # Show first 10
            print(f"  - {img}")
        if len(missing_image_names) > 10:
            print(f"  ... and {len(missing_image_names) - 10} more")
    
    return unified_df
    """Push the dataset to HuggingFace Hub."""
    print(f"\nPushing dataset to {dataset_name}...")
    
    try:
        dataset.push_to_hub(
            dataset_name,
            private=private,
            token=True  # Use the token from huggingface-cli login
        )
        print(f"Successfully pushed dataset to {dataset_name}")
        
        if private:
            print("Dataset is set to private. You can change this in the HuggingFace Hub settings.")
            
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Make sure you're logged in with: huggingface-cli login")

def push_to_hub(dataset, dataset_name: str, private: bool = True):
    """Push the dataset to HuggingFace Hub."""
    print(f"\nPushing dataset to {dataset_name}...")
    
    try:
        dataset.push_to_hub(
            dataset_name,
            private=private,
            token=True  # Use the token from huggingface-cli login
        )
        print(f"Successfully pushed dataset to {dataset_name}")
        
        if private:
            print("Dataset is set to private. You can change this in the HuggingFace Hub settings.")
            
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Make sure you're logged in with: huggingface-cli login")

def main():
    parser = argparse.ArgumentParser(description='Analyze and merge SWE-bench datasets')
    parser.add_argument('--push', action='store_true', 
                       help='Push the unified dataset to HuggingFace Hub')
    parser.add_argument('--public', action='store_true',
                       help='Make the dataset public (default: private)')
    parser.add_argument('--check-images', action='store_true',
                       help='Check if Docker images exist and update image_name field')
    parser.add_argument('--docker-org', default='jyangballin',
                       help='Docker organization name (default: jyangballin)')
    parser.add_argument('--test-image', type=str,
                       help='Test transformation for a specific image name')
    parser.add_argument('--limit-images', type=int, default=None,
                       help='Limit number of images to check (for testing)')
    
    args = parser.parse_args()
    
    # Test image transformation if requested
    if args.test_image:
        print(f"Testing transformations for: {args.test_image}")
        transformations = transform_image_name(args.test_image)
        print("Possible transformations:")
        for i, trans in enumerate(transformations, 1):
            print(f"  {i}. {args.docker_org}/{trans}")
        
        # Check if any exist
        username, password = get_docker_hub_login()
        token = get_dockerhub_token(username, password)
        
        print("\nChecking existence:")
        for trans in transformations:
            exists = check_docker_image_exists(args.docker_org, trans, "latest", token)
            status = "✓ EXISTS" if exists else "✗ NOT FOUND"
            print(f"  {status}: {args.docker_org}/{trans}")
        return
    
    # Load datasets
    trajectories_dataset, smith_dataset = load_datasets()
    
    # Analyze overlap
    common_ids, trajectories_df, smith_df = analyze_overlap(trajectories_dataset, smith_dataset)
    
    if len(common_ids) == 0:
        print("No common instance_ids found. Cannot create unified dataset.")
        return
    
    # Create unified dataset
    unified_dataset = create_unified_dataset(trajectories_df, smith_df, common_ids)
    
    # Convert to DataFrame for image checking
    unified_df = unified_dataset.to_pandas()
    
    # Check Docker images if requested
    if args.check_images:
        unified_df = check_and_update_image_names(unified_df, args.docker_org, args.limit_images)
        unified_dataset = Dataset.from_pandas(unified_df)
    
    # Save locally
    print(f"\nSaving unified dataset locally...")
    unified_dataset.save_to_disk("./swe_smith_unified")
    print("Dataset saved to ./swe_smith_unified")
    
    # Optionally push to hub
    if args.push:
        dataset_name = "SWE-bench/SWE-smith-trajectories-5k-unified"
        push_to_hub(unified_dataset, dataset_name, private=not args.public)
    else:
        print(f"\nTo push to HuggingFace Hub, run:")
        print(f"python {__file__} --push")
        print(f"Add --public flag to make it public")
        print(f"Add --check-images flag to verify Docker images exist")
        print(f"Add --test-image <image_name> to test transformations for a specific image")
        print(f"Add --limit-images N to check only first N images (for testing)")

if __name__ == "__main__":
    main()
