import subprocess
import concurrent.futures
import glob
import pandas as pd
import os
import anndata
import scipy.io
import numpy as np
from pathlib import Path
import logging
import signal
import time

timestamp = time.strftime("%d_%m_%Y,%H:%M")

# Define paths
BASE_DIR = Path("/mnt/LaCIE/skolla")
MAPPED_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/mapped_files"
DOWNLOAD_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/fastq_files"
H5AD_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/h5ad_files"
LOG_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/logs"
WHITELIST_DIR = BASE_DIR / "sc-heart-consortium/ncbi-sra/whitelist_files"

STAR_PATH = "/usr/local/bin/STAR"
STAR_INDEX = BASE_DIR / "dmd_dc_skeletal/index_files/human/"
METADATA_FILE = BASE_DIR / "Github/Heart_Cellular_Collectives_Lopez_Kolla/ncbi_sra/data/Data-Litnukova_Kanemaru_missing.tsv"
FTP_LINKS_FILE = BASE_DIR / "Github/Heart_Cellular_Collectives_Lopez_Kolla/ncbi_sra/data/Data-Litnukova_Kanemaru-links.tsv"

# Configure logging
LOG_FILE = LOG_DIR / "Data-Litnukova_Kanemaru_without_parallel_process.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

# Function to handle subprocess calls and SIGPIPE
def safe_subprocess(command, log_message="Running command"):
    try:
        logging.info(log_message)
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setpgrp)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e.stderr.decode()}")
        return None
    except BrokenPipeError:
        logging.error("Broken pipe error (SIGPIPE) occurred in subprocess.")
        return None

# Function to download FASTQ files
def download_fastqs(run_id, ftp_links):
    output_dir = DOWNLOAD_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for link in ftp_links.split(';'):
        try:
            # Add ftp:// prefix if not present
            if not link.startswith('ftp://'):
                link = 'ftp://' + link.strip()
            
            logging.info(f"Downloading {link} for {run_id}...")
            output_file = output_dir / Path(link).name
            
            # Skip if file already exists and has size > 0
            if output_file.exists() and output_file.stat().st_size > 0:
                logging.info(f"File {output_file} already exists, skipping download...")
                continue
                
            # Try axel first
            axel_command = ['axel', '-n', '12', '-o', str(output_file), link]
            axel_result = safe_subprocess(axel_command, f"Downloading with axel: {link}")
            
            # If axel fails, try wget as fallback
            if not axel_result:
                logging.info(f"Axel failed, trying wget for {link}...")
                wget_command = ['wget', '-O', str(output_file), link]
                wget_result = safe_subprocess(wget_command, f"Downloading with wget: {link}")
                
                if not wget_result:
                    logging.error(f"Both axel and wget failed to download {link}")
                    continue
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logging.info(f"Successfully downloaded {link} for {run_id}")
            else:
                logging.error(f"Download appeared to succeed but file is empty or missing: {output_file}")
                
        except Exception as e:
            logging.error(f"Error downloading {link} for {run_id}: {str(e)}")


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
    output_dir = MAPPED_DIR / run_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    whitelist_filename = WHITELIST_MAPPING.get(assay)

    star_command = [
        STAR_PATH, "--runThreadN", "48",
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
    
    # Run STAR mapping
    if safe_subprocess(star_command, f"Running STAR mapping for {run_id}"):

        expected_output = MAPPED_DIR / run_id / "outputSolo.out" / "GeneFull" / "raw" / "barcodes.tsv"
        
        if expected_output.exists():
            logging.info(f"STAR output found. Deleting FASTQ files for {run_id}...")
            for fastq_file in fastq_dir.glob("*.fastq.gz"):
                try:
                    fastq_file.unlink()
                    logging.info(f"Deleted {fastq_file}")
                except Exception as delete_error:
                    logging.error(f"Could not delete {fastq_file}: {delete_error}")

                            # Delete .sam files
            for sam_file in (MAPPED_DIR / run_id).glob("*.sam"):
                try:
                    sam_file.unlink()
                    logging.info(f"Deleted {sam_file}")
                except Exception as delete_error:
                    logging.error(f"Could not delete {sam_file}: {delete_error}")
        else:
            logging.warning(f"Expected STAR output not found for {run_id} at {expected_output}. Retaining FASTQ files.")
        
        return True
    else:
        logging.error(f"STAR mapping failed for {run_id}. Retaining FASTQ files.")
        return False
    
# Function to convert STAR output to H5AD
def convert_star_to_h5ad(run_accession):
    output_dir = MAPPED_DIR / run_accession / "outputSolo.out" / "GeneFull" / "raw"
    barcodes_file = output_dir / "barcodes.tsv"
    features_file = output_dir / "features.tsv"
    matrix_file = output_dir / "matrix.mtx"
    h5ad_file = H5AD_DIR / f"{run_accession}_GeneFull_{timestamp}.h5ad"
    fastq_dir = DOWNLOAD_DIR / run_accession  

    # Check if all required files exist
    if not all(os.path.exists(f) for f in [barcodes_file, features_file, matrix_file]):
        logging.error(f"Required files are missing for {run_accession}. Skipping conversion.")
        return False

    # Remove existing H5AD file if it exists
    if os.path.exists(h5ad_file):
        os.remove(h5ad_file)

    try:
        logging.info(f"Converting STAR output to H5AD for {run_accession}...")

        # Read and preprocess barcodes
        barcodes = pd.read_csv(barcodes_file, header=None, index_col=0)
        barcodes.index = barcodes.index.astype(str)
        barcodes.index.name = None

        # Read and preprocess features
        features = pd.read_csv(features_file, sep='\t', names=['gene_ids', 'gene_names', 'gene_types'], index_col=1)
        features.index = features.index.astype(str)
        features.index.name = None
        features.columns = features.columns.astype(str)

        # Load the matrix file and ensure it’s a sparse CSR matrix
        matrix = scipy.io.mmread(matrix_file)
        if not isinstance(matrix, scipy.sparse.csr_matrix):
            matrix = scipy.sparse.csr_matrix(matrix.T)  # Transpose and convert to CSR for AnnData compatibility

        # Verify matrix dimensions match barcodes and features
        if matrix.shape[0] != len(barcodes) or matrix.shape[1] != len(features):
            logging.error(f"Dimension mismatch: Matrix shape {matrix.shape}, Barcodes {len(barcodes)}, Features {len(features)}")
            return False

        # Create AnnData object, checking for correct data types and structure
        adata = anndata.AnnData(X=matrix, obs=barcodes, var=features)
        adata.layers['spliced'] = matrix
        adata.var_names_make_unique()

       # Create AnnData object 
        adata.write(h5ad_file)
        logging.info(f"Successfully converted to H5AD: {h5ad_file}")

        return True
    except Exception as e:
        logging.error(f"Error converting STAR output to H5AD for {run_accession}: {e}")
        if hasattr(e, 'args') and e.args:
            logging.error(f"Details: {e.args[0]}")
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

    # Define expected STAR output file path
    star_output_dir = MAPPED_DIR / run_accession
    expected_star_file = os.path.join(star_output_dir, "outputLog.final.out") 

    if os.path.exists(expected_star_file):
        logging.info(f"STAR output file found for {run_accession}, skipping download and mapping.")
    else:
        # Download FASTQ files
        download_fastqs(run_accession, run_info['submitted_ftp'])

        # Map with STAR
        assay = run_info['Assay']
        if run_star(run_accession, assay):
            logging.info(f"STAR mapping completed for {run_accession}")
        else:
            logging.warning(f"STAR mapping failed for {run_accession}")
        
    # Convert to H5AD if STAR output exists 
    convert_star_to_h5ad(run_accession)

# Function to run processes in an order
def process_run(run_info):
    run_accession = run_info['Run']
    logging.info(f"Processing run: {run_accession}")

    # Set up logging for this run
    run_logger = setup_run_logger(run_accession)

    try:
        # Define expected STAR output file path
        star_output_dir = MAPPED_DIR / run_accession / "outputSolo.out" / "GeneFull" / "raw"
        expected_star_file = os.path.join(star_output_dir, "barcodes.tsv")

        if os.path.exists(expected_star_file):
            logging.info(f"STAR output file found for {run_accession}, skipping download and mapping.")
        else:
            # Check if assay type is supported
            assay = run_info['Assay']
            if assay not in ASSAY_PARAMS:
                logging.error(f"Unsupported assay type for {run_accession}: {assay}")
                return

            # Download FASTQ files
            download_fastqs(run_accession, run_info['submitted_ftp'])

            # Map with STAR
            if run_star(run_accession, assay):
                logging.info(f"STAR mapping completed for {run_accession}")
            else:
                logging.error(f"STAR mapping failed for {run_accession}")
                return

        # Convert to H5AD if STAR output exists
        if convert_star_to_h5ad(run_accession):
            logging.info(f"H5AD conversion completed for {run_accession}")
        else:
            logging.error(f"H5AD conversion failed for {run_accession}")

    except Exception as e:
        logging.error(f"Error in process_run for {run_accession}: {str(e)}")
        raise


def main():
    try:
        # Read metadata and submitted_ftp
        metadata = pd.read_csv(METADATA_FILE, sep="\t")
        ftp_links = pd.read_csv(FTP_LINKS_FILE, sep="\t")
        
        # Diagnostic logging for input data
        logging.info(f"Total rows in metadata: {len(metadata)}")
        logging.info(f"Total rows in ftp_links: {len(ftp_links)}")
        
        # Check for any missing or null values in key columns
        logging.info(f"Null values in metadata 'Run' column: {metadata['Run'].isnull().sum()}")
        logging.info(f"Null values in ftp_links 'Run' column: {ftp_links['Run'].isnull().sum()}")

        # Merge metadata with submitted_ftp on run accession
        merged_info = pd.merge(metadata, ftp_links, on='Run', how='left')
        logging.info(f"Total rows after merging: {len(merged_info)}")
        
        # Log information about any runs that didn't get FTP links
        missing_ftp = merged_info[merged_info['submitted_ftp'].isnull()]
        if not missing_ftp.empty:
            logging.warning(f"Runs missing FTP links: {missing_ftp['Run'].tolist()}")

        # Counter for processed samples
        processed_count = 0

        # Sequential processing of each run
        for idx, run_info in merged_info.iterrows():
            try:
                logging.info(f"Starting to process run {idx + 1} of {len(merged_info)}: {run_info['Run']}")
                
                if pd.isnull(run_info['submitted_ftp']):
                    logging.error(f"Skipping {run_info['Run']} due to missing FTP link")
                    continue
                
                if pd.isnull(run_info['Assay']):
                    logging.error(f"Skipping {run_info['Run']} due to missing Assay information")
                    continue

                # Process the run
                process_run(run_info)
                processed_count += 1
                logging.info(f"Successfully processed {processed_count} samples so far")

            except Exception as e:
                logging.error(f"Error processing run {run_info['Run']}: {e}")

        logging.info(f"Processing completed. Total samples processed: {processed_count}")

    except Exception as e:
        logging.error(f"An error occurred in the main processing loop: {e}")
        raise

if __name__ == "__main__":
    main()
