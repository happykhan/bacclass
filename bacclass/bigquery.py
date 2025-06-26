#!/usr/bin/env python3
"""
Fetch biosample information from BigQuery public datasets
"""

import sys

try:
    from google.cloud import bigquery
except ImportError as e:
    print(f"Error: Google Cloud BigQuery package not installed: {e}")
    print("Install with: pip install google-cloud-bigquery")
    sys.exit(1)



def fetch_species_biosamples(species, limit=10, output_file=None):
    """
    Fetch multiple Species biosamples from BigQuery
    
    Args:
        limit (int): Number of records to fetch (default: 10)
        output_file (str): Optional CSV file to save results
    
    Returns:
        list: List of biosample dictionaries
    """
    try:
        client = bigquery.Client()
    except Exception as e:
        print(f"Error setting up BigQuery client: {e}")
        print("Make sure you have:")
        print("1. Google Cloud SDK installed")
        print("2. Authentication set up: gcloud auth application-default login")
        print("3. A valid Google Cloud project configured")
        return []
        
    # Query to get Salmonella records with required columns
    query_sql = f"""
    SELECT
      acc,
      center_name,
      biosample,
      organism,
      bioproject,
      geo_loc_name_country_calc,
      geo_loc_name_country_continent_calc,
      geo_loc_name_sam,
      attributes
    FROM `nih-sra-datastore.sra.metadata`
    WHERE organism LIKE '%{species}%'
      AND center_name != 'SC'
      AND center_name != 'THE WELLCOME TRUST SANGER INSTITUTE'
      AND center_name != 'WELLCOME SANGER INSTITUTE'
    LIMIT {limit}
    """
    
    print(f"Querying BigQuery for {limit} {species} records...")
    
    try:
        query_job = client.query(query_sql)
        results = list(query_job)
    except Exception as e:
        print(f"Error executing BigQuery: {e}")
        return []
    
    if not results:
        print(f"No {species} data found")
        return []
    
    processed_records = []
    
    for row in results:
        # Start with the basic columns
        record = {
            'acc': row.acc,
            'center_name': row.center_name,
            'biosample': row.biosample,
            'organism': row.organism,
            'bioproject': row.bioproject,
            'geo_loc_name_country_calc': row.geo_loc_name_country_calc,
            'geo_loc_name_country_continent_calc': row.geo_loc_name_country_continent_calc,
            'geo_loc_name_sam': ';'.join(row.geo_loc_name_sam) if isinstance(row.geo_loc_name_sam, list) else row.geo_loc_name_sam
        }
        
        # Process attributes, excluding those with key "primary_search"
        if row.attributes and hasattr(row.attributes, '__iter__'):
            for attr in row.attributes: # # type: ignore 
                key = attr.get('k', '') or attr.get('name', '')
                value = attr.get('v', '') or attr.get('value', '')
                
                # Skip attributes with key "primary_search"
                if key and key != 'primary_search':
                    record[f'attr_{key}'] = value
        
        processed_records.append(record)
    
    print(f"Successfully processed {len(processed_records)} records")
    
    # Save to CSV if requested
    if output_file:
        try:
            import pandas as pd
            
            df = pd.DataFrame(processed_records)
            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
            
        except ImportError:
            print("pandas not available for CSV export")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    return processed_records    

def subsample_records(processed_records, limit=1000, output_file=None):
    """
    Create a subsample of records, evenly sampling across different 'center_name' values.
    
    Args:
        processed_records (list): List of record dictionaries
        limit (int): Target number of records in the subsample
        output_file (str): Optional CSV file to save the subsample
    
    Returns:
        list: Subsampled records
    """
    if not processed_records:
        print("No records to subsample")
        return []
    
    if len(processed_records) <= limit:
        print(f"Number of records ({len(processed_records)}) is already <= limit ({limit})")
        if output_file:
            try:
                import pandas as pd
                df = pd.DataFrame(processed_records)
                df.to_csv(output_file, index=False)
                print(f"All records saved to {output_file}")
            except Exception as e:
                print(f"Error saving to CSV: {e}")
        return processed_records
    
    # Group records by center_name
    from collections import defaultdict
    records_by_center = defaultdict(list)
    
    for record in processed_records:
        center = record.get('center_name', 'unknown')
        records_by_center[center].append(record)
    
    # Display center distribution
    print(f"Found {len(records_by_center)} unique centers:")
    for center, records in sorted(records_by_center.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {center}: {len(records)} records")
    
    # Calculate how many records to take from each center
    num_centers = len(records_by_center)
    base_per_center = limit // num_centers
    remainder = limit % num_centers
    
    print(f"\nSubsampling strategy:")
    print(f"  Target total: {limit} records")
    print(f"  Number of centers: {num_centers}")
    print(f"  Base per center: {base_per_center}")
    print(f"  Extra records to distribute: {remainder}")
    
    subsampled_records = []
    centers_sorted = sorted(records_by_center.keys())
    
    for i, center in enumerate(centers_sorted):
        center_records = records_by_center[center]
        
        # Calculate how many to take from this center
        records_to_take = base_per_center
        if i < remainder:  # Distribute remainder across first few centers
            records_to_take += 1
        
        # Don't take more than available
        records_to_take = min(records_to_take, len(center_records))
        
        # Randomly sample from this center's records
        import random
        sampled = random.sample(center_records, records_to_take)
        subsampled_records.extend(sampled)
        
        print(f"  {center}: took {records_to_take} from {len(center_records)} available")
    
    print(f"\nSubsampling completed: {len(subsampled_records)} records selected")
    
    # Shuffle the final result to mix centers
    import random
    random.shuffle(subsampled_records)
    
    # Save to CSV if requested
    if output_file:
        try:
            import pandas as pd
            df = pd.DataFrame(subsampled_records)
            df.to_csv(output_file, index=False)
            print(f"Subsampled data saved to {output_file}")
        except ImportError:
            print("pandas not available for CSV export")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    return subsampled_records
