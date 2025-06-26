"""
Bacterial Biosample Classifier

This module classifies bacterial biosample information from NCBI or ENA databases
into Clinical, Environmental, or Food categories using OpenAI's ChatGPT API.
"""

import re
import json
import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_prompt(biosample_info: str) -> str: 
    classification_prompt = f"""
    You are an expert microbiologist specializing in Salmonella classification. Your task is to classify Salmonella biosample records into one of 11 categories based on their isolation source and context:

    1. **Clinical**: Samples isolated from human or animal patients, clinical specimens, hospitals, medical facilities
    2. **Environmental**: Samples from environmental sources like water, soil, sewage, wildlife, natural habitats
    3. **Food**: Samples from food products, food processing facilities, food production environments
    4. **Companion Animal**: Samples from pets or companion animals, veterinary clinical samples
    5. **Aquatic animal**: Samples from aquatic animals
    6. **Animal feed**: Samples from animal feed or feed production environments
    7. **Laboratory**: Samples from laboratory settings, research environments, or experimental setups
    8. **Livestock**: Samples from livestock or farm animals
    9. **ND**: Not determined or not applicable
    10. **Poultry**: Samples from poultry or poultry production environments
    11. **Wild animal**: Samples from wild animals or wildlife habitats

    Please analyze the following Salmonella biosample information and classify it into exactly one category. Consider all available information including isolation source, host, sample type, location, and any other relevant details.

    Biosample Information:
    {biosample_info}

    Respond with only a JSON object in this exact format:
    {{
        "classification": "One of: Clinical, Environmental, Food, Companion Animal, Aquatic animal, Animal feed, Laboratory, Livestock, ND, Poultry, Wild animal",
        "confidence": "High,Medium,Low",
        "reasoning": "Brief explanation of your classification decision",
        "category_number": "1-11 (the number corresponding to the category above)"
    }}

    Classification: """
    return classification_prompt

def classify_csv(csv_path, output_path):
    """
    Classify biosamples from a CSV file using OpenAI API.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_path (str): Path to save classification results (optional)
    
    Returns:
        list: Classification results
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    results = []
    
    logger.info(f"Starting classification of {len(df)} records from {csv_path}")
    
    for idx, (index, row) in enumerate(df.iterrows()):
        logger.info(f"Processing record {idx + 1}/{len(df)}")
        
        # Convert row to biosample info string
        # Ignore certain technical columns that don't help with classification
        columns_to_ignore = [
            'attr_bases', 'attr_bytes', 'attr_run_file_create_date', 
            'attr_ena_first_public_sam', 'attr_collection_date_sam', 
            'attr_ena_last_update_sam', 'attr_external_id_sam',
            'attr_insdc_first_public_sam', 'attr_insdc_last_update_sam', 
            'attr_insdc_status_sam'
        ]
        
        # Create a copy of the row and drop unwanted columns
        filtered_row = row.copy()
        for col in columns_to_ignore:
            if col in filtered_row.index:
                filtered_row = filtered_row.drop(col)
        
        # Create biosample info string
        biosample_info = ", ".join(
            f"{key}: {value}" for key, value in filtered_row.items() 
            if pd.notna(value) and str(value).lower() not in ["", 'missing', 'unknown', 'not provided']
        )
        
        
        try:
            client = OpenAI()

            response = client.responses.create(
                prompt={
                    "id": "pmpt_685d19514a808197be55712806f40c520a033e61d6863fbb",
                    "version": "3",
                    "variables": {
                    "biosample_info": biosample_info
                    }
                }
            )                        
            # Parse the response
            response_content = response.output_text
            if response_content is None:
                raise ValueError("Received empty response from OpenAI")
            
            response_text = response_content.strip()
            logger.debug(f"Raw response: {response_text}")
            
            # Try to extract JSON from the response
            try:
                classification = json.loads(response_text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Add the original accession/identifier
            result = row.to_dict() | {
                "CLASSIFICATION": classification.get('classification', 'Unknown'),
                "CONFIDENCE": classification.get('confidence', 'Unknown'),
                "REASONING": classification.get('reasoning', 'No reasoning provided'),
                "CATEGORY_NUMBER": classification.get('category_number', 'Unknown')
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error classifying record {idx}: {e}")
            # Add a failed classification result
            result = row.to_dict()
            result = {
                "CLASSIFICATION": "Error",
                "CONFIDENCE": "Unknown",
                "REASONING": f"Classification failed: {str(e)}",
                "CATEGORY_NUMBER": "Unknown"
            }
            results.append(result)
    
    logger.info(f"Classification completed. {len(results)} results generated.")
    
    # Save results if output path is provided
    if output_path:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Classification results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    classify_csv(
        csv_path="classification_data/subsampled_salmonella_records.csv", 
        output_path="classification_data/classification_results.csv"
    )
    