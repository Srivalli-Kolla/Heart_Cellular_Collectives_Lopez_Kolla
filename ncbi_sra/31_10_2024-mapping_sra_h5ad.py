import subprocess
import glob
import pandas as pd
import os
import anndata
import scipy.io
import numpy as np
from pathlib import Path
import logging

# Define paths
BASE_DIR = Path("/mnt/LaCIE/skolla")
MAPPED_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/mapped_files"
DOWNLOAD_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/fastq_files"
H5AD_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/h5ad_files"
LOG_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/logs"
WHITELIST_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/whitelist_files"

STAR_PATH = "/usr/local/bin/STAR"
STAR_INDEX = BASE_DIR / "dmd_dc_skeletal/index_files/human/"
METADATA_FILE = BASE_DIR / "Github/Heart_Cellular_Collectives_Lopez_Kolla/ncbi_sra/data/Data-D3_litnukova.tsv"
FTP_LINKS_FILE = BASE_DIR / "Github/Heart_Cellular_Collectives_Lopez_Kolla/ncbi_sra/data/Data-D3_litnukova-links.tsv"

# Define whitelist mapping for assay types
WHITELIST_MAPPING = {
    "Single Cell 3' v1": "737K-april-2014_rc.txt",
    "Single Cell 3' v2": "737K-august-2016.txt",
    "Single Cell 3' v3": "3M-february-2018.txt",
    "Single Cell 5' v1.1": "737K-august-2016.txt",
    "Single Cell 5' v2": "737K-august-2016.txt",
    "Single Cell 5' v3": "737K-august-2016.txt",
    "Single Cell Multiome v1": "737K-arc-v1.txt",
}

# STAR parameters for each technology type
ASSAY_PARAMS = {
    "Single Cell 3' v1": {"cb_len": 14, "umi_len": 10, "strand": "Forward"},
    "Single Cell 3' v2": {"cb_len": 16, "umi_len": 10, "strand": "Forward"},
    "Single Cell 3' v3": {"cb_len": 16, "umi_len": 12, "strand": "Forward"},
    "Single Cell 5' v1.1": {"cb_len": 14, "umi_len": 10, "strand": "Reverse"},
    "Single Cell 5' v2": {"cb_len": 14, "umi_len": 10, "strand": "Reverse"},
    "Single Cell 5' v3": {"cb_len": 16, "umi_len": 12, "strand": "Reverse"},
    "Single Cell Multiome v1": {"cb_len": 16, "umi_len": 12, "strand": "Forward"},
}

# Configure logging
LOG_FILE = LOG_DIR / "Data-D3_litnukova.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to download FASTQ files
def download_fastqs(run_id, ftp_links):
    output_dir = DOWNLOAD_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for link in ftp_links.split(';'):
        link = link.strip()  # Clean any extra whitespace
        try:
            logging.info(f"Downloading {link} for {run_id}...")
            output_file = output_dir / Path(link).name
            
            # Use axel to download with multiple connections
            axel_command = ['axel', '-n', '8', '-o', str(output_file), link]
            subprocess.run(axel_command, check=True)
            logging.info(f"Successfully downloaded {link} for {run_id}.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to download {link} for {run_id}: {e}. Retrying...")
            try:
                subprocess.run(axel_command, check=True)
                logging.info(f"Successfully downloaded {link} for {run_id} on retry.")
            except subprocess.CalledProcessError as retry_e:
                logging.error(f"Retry failed for {link} for {run_id}: {retry_e}")
        except Exception as general_e:
            logging.error(f"An unexpected error occurred for {link} for {run_id}: {general_e}")

# Function to run STAR mapping
def run_star(run_id, assay):
    fastq_dir = DOWNLOAD_DIR / run_id
    fastq_files = sorted(fastq_dir.glob("*.fastq.gz"))

    fastq_files_r1 = [file for file in fastq_files if "_R1_" in file.name]
    fastq_files_r2 = [file for file in fastq_files if "_R2_" in file.name]

    if not fastq_files_r1 or not fastq_files_r2:
        logging.error(f"FASTQ files missing for {run_id}.")
        return False

    assay_params = ASSAY_PARAMS.get(assay)
    if not assay_params:
        logging.error(f"No STAR parameters available for assay {assay}")
        return False

    output_dir = MAPPED_DIR / run_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)  

    whitelist_filename = WHITELIST_MAPPING.get(assay)
    if not whitelist_filename:
        logging.error(f"No whitelist file found for assay type {assay}")
        return False

    star_command = [
        STAR_PATH, "--runThreadN", "32",
        "--genomeDir", str(STAR_INDEX),
        "--readFilesIn", str(fastq_files_r2[0]), str(fastq_files_r1[0]),
        "--runDirPerm", "All_RWX",
        "--soloCBwhitelist", str(WHITELIST_DIR / whitelist_filename),
        "--soloType", "CB_UMI_Simple",
        "--soloCBstart", "1",
        "--soloCBlen", str(assay_params["cb_len"]),
        "--soloUMIstart", str(assay_params["cb_len"] + 1),
        "--soloUMIlen", str(assay_params["umi_len"]),
        "--soloStrand", assay_params["strand"],
        "--soloFeatures", "Gene", "GeneFull",
        "--readFilesCommand", "zcat",
        "--outFileNamePrefix", str(output_dir / "")
    ]

    try:
        subprocess.run(star_command, check=True)
        logging.info(f"STAR mapping completed for {run_id}.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"STAR mapping failed for {run_id} with error: {e}")
        return False
    
# Function to convert STAR output to H5AD
def convert_star_to_h5ad(run_accession):
    output_dir = MAPPED_DIR / run_accession / "outputSolo.out" / "GeneFull" / "raw"
    barcodes_file = output_dir / "barcodes.tsv"
    features_file = output_dir / "features.tsv"
    matrix_file = output_dir / "matrix.mtx"
    h5ad_file = H5AD_DIR / f"{run_accession}.h5ad"

    if not all(os.path.exists(f) for f in [barcodes_file, features_file, matrix_file]):
        logging.error(f"Required files are missing for {run_accession}. Skipping conversion.")
        return False

    # Remove existing H5AD file to allow overwriting
    if os.path.exists(h5ad_file):
        os.remove(h5ad_file)

    try:
        logging.info(f"Converting STAR output to H5AD for {run_accession}...")
        barcodes = pd.read_csv(barcodes_file, header=None)
        features = pd.read_csv(features_file, header=None, sep='\t', names=['gene_id', 'gene_name'])
        matrix = scipy.io.mmread(matrix_file).tocsr()

        adata = anndata.AnnData(X=matrix.T, obs=pd.DataFrame(index=barcodes[0]), var=features.set_index('gene_id'))
        adata.write(h5ad_file)
        logging.info(f"Successfully converted to H5AD: {h5ad_file}")

        return True
    except Exception as e:
        logging.error(f"Error converting STAR output to H5AD for {run_accession}: {e}")
        return False

# Function to configure a run-specific logger
def setup_run_logger(run_accession):
    run_log_file = LOG_DIR / f"{run_accession}.log"
    run_logger = logging.getLogger(run_accession)
    run_logger.setLevel(logging.INFO)
    
    # Create file handler for run-specific log file
    file_handler = logging.FileHandler(run_log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    run_logger.addHandler(file_handler)
    
    return run_logger

def process_run(run_info):
    run_accession = run_info['Run']
    logging.info(f"Processing run: {run_accession}")

    # Set up logging for this run
    run_logger = setup_run_logger(run_accession)

    # Download FASTQ files
    download_fastqs(run_accession, run_info['submitted_ftp'])

    # Map with STAR
    assay = run_info['Assay']
    if run_star(run_accession, assay):
        # Convert to H5AD if STAR mapping was successful
        convert_star_to_h5ad(run_accession)

def main():
    try:
        # Read metadata and submitted_ftp
        metadata = pd.read_csv(METADATA_FILE, sep="\t")
        ftp_links = pd.read_csv(FTP_LINKS_FILE, sep="\t")

        # Merge metadata with submitted_ftp on run accession
        merged_info = pd.merge(metadata, ftp_links, on='Run', how='left')

        # Process each run
        for _, run_info in merged_info.iterrows():
            process_run(run_info)
    except Exception as e:
        logging.error(f"An error occurred in the main processing loop: {e}")

if __name__ == "__main__":
    main()