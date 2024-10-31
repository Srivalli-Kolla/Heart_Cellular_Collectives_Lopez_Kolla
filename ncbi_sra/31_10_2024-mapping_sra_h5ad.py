import subprocess
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
METADATA_FILE = BASE_DIR / "Github/Heart_Cellular_Collectives_Lopez_Kolla/ncbi_sra/data/Data-Litnukova_Kanemaru.tsv"
FTP_LINKS_FILE = BASE_DIR / "Github/Heart_Cellular_Collectives_Lopez_Kolla/ncbi_sra/data/Data-Litnukova_Kanemaru-links.tsv"

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
LOG_FILE = LOG_DIR / "Data-Litnukova_Kanemaru.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to download fastqs
def download_fastq(run_id, ftp_links):
    """Download FASTQ files using axel."""
    output_dir = DOWNLOAD_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for link in ftp_links.split(';'):
        try:
            logging.info(f"Downloading {link} for {run_id}...")
            axel_command = ['axel', '-n', '8', '-o', str(output_dir), link]
            subprocess.run(axel_command, check=True)
            logging.info(f"Successfully downloaded {link} for {run_id}.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to download {link} for {run_id}: {e}")

# Function to run STAR alignmnet
def run_star(run_id, assay):
    """Run STAR for mapping based on the assay type."""
    fastq_dir = DOWNLOAD_DIR / run_id
    fastq_files = sorted([f for f in fastq_dir.glob("*.fastq.gz")])

    fastq_files_r1 = [file for file in fastq_files if "_R1_" in file.name]
    fastq_files_r2 = [file for file in fastq_files if "_R2_" in file.name]

    if not fastq_files_r1 or not fastq_files_r2:
        logging.error(f"FASTQ files missing for {run_id}.")
        return  

    assay_params = ASSAY_PARAMS.get(assay)
    if not assay_params:
        logging.error(f"No STAR parameters available for assay {assay}")
        return

    output_dir = MAPPED_DIR / run_id / "output"
    output_dir.mkdir(parents=True, exist_ok=True)  

    whitelist_filename = WHITELIST_MAPPING.get(assay)
    if not whitelist_filename:
        logging.error(f"No whitelist file found for assay type {assay}")
        return

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
    except subprocess.CalledProcessError as e:
        logging.error(f"STAR mapping failed for {run_id} with error: {e}")

# Function to convert STAR outout to h5ad
def convert_to_h5ad(run_id):
    """Convert STAR output to H5AD format."""
    
    matrix_dir = MAPPED_DIR / run_id / "outputSolo.out" / "GeneFull" / "raw"
    logging.info(f"Checking matrix directory: {matrix_dir}")
    
    if not matrix_dir.exists():
        logging.error(f"Matrix directory does not exist for {run_id}: {matrix_dir}")
        return

    matrix_file = matrix_dir / "matrix.mtx"
    barcodes_file = matrix_dir / "barcodes.tsv"
    features_file = matrix_dir / "features.tsv"

    if not matrix_file.exists():
        logging.error(f"Matrix file missing for {run_id}: {matrix_file}")
        return
    if not barcodes_file.exists():
        logging.error(f"Barcodes file missing for {run_id}: {barcodes_file}")
        return
    if not features_file.exists():
        logging.error(f"Features file missing for {run_id}: {features_file}")
        return

    try:
       
        mtx = scipy.io.mmread(matrix_file).tocsc()

       
        barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")[0].values
        features = pd.read_csv(features_file, header=None, sep="\t")[1].values

        if mtx.shape[0] != len(features) or mtx.shape[1] != len(barcodes):
            logging.error(f"Matrix dimensions {mtx.shape} do not match the lengths of barcodes {len(barcodes)} or features {len(features)}.")
            return

    
        adata = anndata.AnnData(X=mtx, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=features))

        
        output_path = H5AD_DIR / f"{run_id}.h5ad"

        
        adata.write(output_path)
        logging.info(f"Converted STAR output to H5AD format for {run_id} at {output_path}")
    except Exception as e:
        logging.error(f"Failed to convert to H5AD for {run_id} with error: {e}")

# Function to run all necessary steps
def process_run(run_id, assay, ftp_links):
    try:
        # Define paths
        h5ad_path = H5AD_DIR / f"{run_id}.h5ad"
        matrix_dir = MAPPED_DIR / run_id / "outputSolo.out" / "GeneFull" / "raw"
        fastq_dir = DOWNLOAD_DIR / run_id
        
        # Check if STAR output exists and convert to H5AD if found
        if matrix_dir.exists():
            logging.info(f"STAR output found for {run_id}. Converting to H5AD.")
            convert_to_h5ad(run_id)
            return
        
        # Check for existing FASTQ files
        fastq_files_r1 = list(fastq_dir.glob("*_R1_*.fastq.gz"))
        fastq_files_r2 = list(fastq_dir.glob("*_R2_*.fastq.gz"))

        if fastq_files_r1 and fastq_files_r2:
            logging.info(f"FASTQ files found for {run_id}. Running STAR alignment.")
            run_star(run_id, assay)

            # Ensure STAR alignment was successful
            if matrix_dir.exists():  # Only convert if STAR has generated output
                convert_to_h5ad(run_id)
            else:
                logging.error(f"STAR output was not generated for {run_id} after alignment.")
            return

        # If no files are found, download FASTQ files
        logging.info(f"No files found for {run_id}. Downloading FASTQ files.")
        download_fastq(run_id, ftp_links)
        
        # Run STAR and convert to H5AD after downloading
        run_star(run_id, assay)

        # Check STAR output again after attempting to run
        if matrix_dir.exists():
            convert_to_h5ad(run_id)
        else:
            logging.error(f"STAR output was not generated for {run_id} after alignment.")

    except Exception as e:
        logging.error(f"Error processing {run_id} during conversion: {e}")

# Function to perform tasks
def main():
    
    MAPPED_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    H5AD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    
    metadata = pd.read_csv(METADATA_FILE, sep="\t")
    ftp_links = pd.read_csv(FTP_LINKS_FILE, sep="\t")

    
    ftp_links_dict = ftp_links.set_index('Run')['submitted_ftp'].to_dict()

    
    for run_id in metadata['Run'].unique():
        try:
            
            assay = metadata.loc[metadata['Run'] == run_id, 'Assay'].values[0]

            
            ftp_links_for_run = ftp_links_dict.get(run_id)
            if not ftp_links_for_run:
                logging.warning(f"No FTP links found for run ID {run_id}. Skipping.")
                continue

            # Process the current run
            logging.info(f"Starting process for run ID {run_id}, assay {assay}.")
            process_run(run_id, assay, ftp_links_for_run)
            logging.info(f"Completed process for run ID {run_id}.")

        except Exception as e:
            logging.error(f"Error processing run ID {run_id}: {e}")

if __name__ == "__main__":
    main()