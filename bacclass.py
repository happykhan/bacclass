#!/usr/bin/env python3
"""
Bacterial Biosample Classification System CLI

A command-line interface for preparing datasets from BigQuery and running
bacterial biosample classification using OpenAI's ChatGPT API.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from bacclass.setup_bigquery import main as setup_bigquery
from bacclass.test_classifier import evaluate_classification_results
from bacclass.bigquery import fetch_species_biosamples, subsample_records
from bacclass.classifier import classify_csv
try:
    import typer
    import pandas as pd
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install typer rich pandas")
    sys.exit(1)

# Add the bacclass package to the path
sys.path.insert(0, str(Path(__file__).parent))


app = typer.Typer(
    name="bacclass",
    help="Bacterial Biosample Classification System - Extract data from BigQuery and classify using AI",
    add_completion=False,
)

console = Console()

# Global options
def version_callback(value: bool):
    if value:
        typer.echo("BacClass v1.0.0")
        raise typer.Exit()



@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, help="Show version and exit"
    )
):
    """
    Bacterial Biosample Classification System
    
    This tool helps you:
    1. Extract biosample data from BigQuery public datasets
    2. Classify bacterial biosamples using AI (OpenAI ChatGPT)
    """
    pass

@app.command("setup-bigquery")
def setup_bigquery_command():
    """
    Setup BigQuery client and authentication."""
    setup_bigquery()

@app.command("prepare-dataset")
def prepare_dataset(
    species: str = typer.Option(
        "Salmonella", 
        "--species", 
        "-s", 
        help="Species to search for (e.g., 'Salmonella', 'E. coli')"
    ),
    limit: int = typer.Option(
        1000,
        "--limit",
        "-l", 
        help="Maximum number of records to fetch"
    ),
    output_dir: Path = typer.Option(
        "classification_data",
        "--output",
        "-o",
        help="Output directory for dataset files"
    )
):
    """
    Prepare a dataset by extracting biosample data from BigQuery.
    
    This command fetches bacterial biosample records from the NCBI SRA 
    BigQuery public dataset, including metadata and attributes needed 
    for classification.
    
    Examples:
        bacclass prepare-dataset --species "Salmonella enterica" --limit 500
    """
    console.print(f"[bold green]Preparing {species} dataset...[/bold green]")

    # Use real BigQuery
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Fetching {limit} {species} records from BigQuery...", total=None)
        os.makedirs(output_dir, exist_ok=True)
        output = Path(output_dir) / f"{species.replace(' ', '_').lower()}_records.csv"
        subsample_output = Path(output_dir) / f"subsampled_{species.replace(' ', '_').lower()}_records.csv"
        try:
            records = fetch_species_biosamples(
                species=species,
                limit=limit,
                output_file=str(output),
                
            )
            progress.update(task, advance=limit)
            console.print(f"[blue]Fetched {len(records)} records for {species}[/blue]")
            # Create subsample
            if records:
                subsample_records(records, limit=1000, output_file=subsample_output)
            
            if not records:
                console.print("[red]No records found or BigQuery authentication failed[/red]")
                console.print("[yellow]Try using --mock-data for testing[/yellow]")
                raise typer.Exit(1)
            
            progress.update(task, completed=True)
            
        except Exception as e:
            console.print(f"[red]Error fetching data: {e}[/red]")
            console.print("[yellow]Tip: Make sure you have Google Cloud authentication set up[/yellow]")
            console.print("[yellow]Run: gcloud auth application-default login[/yellow]")
            raise typer.Exit(1)
    
    # Show summary
    try:
        df = pd.read_csv(output)
        
        table = Table(title=f"Dataset Summary: {output}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Records", str(len(df)))
        table.add_row("Species", species)
        table.add_row("Columns", str(len(df.columns)))
        table.add_row("Output File", str(output))
        
        if 'organism' in df.columns:
            unique_organisms = df['organism'].nunique()
            table.add_row("Unique Organisms", str(unique_organisms))
        
        if 'center_name' in df.columns:
            unique_centers = df['center_name'].nunique()
            table.add_row("Unique Centers", str(unique_centers))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not read output file for summary: {e}[/yellow]")
    
    console.print("[bold green]✓ Dataset preparation completed![/bold green]")

@app.command("run-classifier")
def run_classifier(
    csv_path: Path = typer.Option(
        "classification_data/subsampled_salmonella_records.csv",
        "--csv-path",
        "-c",
        help="Path to the CSV file with subsampled biosample records"
    ),
    output_path: Path = typer.Option(
        "classification_data/classification_results.csv",
        "--output",
        "-o",
        help="Path to save the classification results CSV file"
    ),    
):
    classify_csv(csv_path=csv_path, output_path=output_path)    

@app.command("eval-classifier")
def test_classifier(
    input_file: Path = typer.Option(
        "classification_data/classification_results.csv", 
        "--input", 
        "-i", 
        help="Input CSV file with biosample data",
        exists=True
    ),
    output_dir: Path = typer.Option(
        "evaluation_results",
        "--output",
        "-o",
        help="Output dir for classification results"
    )
):

    # Run evaluation
    try:
        results = evaluate_classification_results(str(input_file), str(output_dir))
        console.print("\n" + "="*60, style="bold blue")
        console.print("EVALUATION SUMMARY", style="bold green")
        console.print("="*60, style="bold blue")
        console.print(f"[cyan]Input file:[/cyan] {input_file}")
        console.print(f"[cyan]Total samples:[/cyan] {results['data_summary']['total_samples']}")
        console.print(f"[cyan]Clean samples:[/cyan] {results['data_summary']['clean_samples']}")
        console.print(f"[cyan]Overall accuracy:[/cyan] [bold]{results['metrics']['accuracy']:.4f}[/bold]")
        console.print(f"[cyan]Macro F1 score:[/cyan] [bold]{results['metrics']['macro_f1']:.4f}[/bold]")
        console.print(f"[cyan]Weighted F1 score:[/cyan] [bold]{results['metrics']['weighted_f1']:.4f}[/bold]")
        console.print(f"[cyan]Cohen's Kappa:[/cyan] [bold]{results['metrics']['cohen_kappa']:.4f}[/bold]")
        console.print(f"\n[green]Results saved to:[/green] {output_dir}")
        console.print("[magenta]Generated files:[/magenta]")
        for plot_name, plot_path in results['plot_paths'].items():
            console.print(f"  - [yellow]{plot_name}[/yellow]: {plot_path}")
        console.print(f"  - [yellow]Report[/yellow]: {results['report_path']}")
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise        


    
if __name__ == "__main__":
    app()
