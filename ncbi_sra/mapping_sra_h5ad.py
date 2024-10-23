import os
import subprocess
import glob
import pandas as pd
import scipy.io
import anndata
from multiprocessing import Pool
import logging

# Define paths
BASE_DIR = "/mnt/LaCIE/skolla"
MAPPED_DIR = os.path.join(BASE_DIR, "sc-heart-consortium/ncbi-sra/mapped_files")
H5AD_DIR = os.path.join(BASE_DIR, "sc-heart-consortium/ncbi-sra/h5ad_files")
STAR_INDEX = os.path.join(BASE_DIR, "dmd_sc/index_files/human/")
LOG_DIR = os.path.join(BASE_DIR, "sc-heart-consortium/ncbi-sra/logs")
WHITELIST_DIR = os.path.join(BASE_DIR, "sc-heart-consortium/ncbi-sra/whitelist_files")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "sc-heart-consortium/ncbi-sra/fastq_files")

STAR_PATH = "/usr/local/bin/STAR"
FASTERQ_DUMP_PATH = "/home/skolla/sratoolkit.3.1.1-ubuntu64/bin/fastq-dump"

METADATA_FILE = '/home/skolla/Github/sc-heart-consortium/ncbi_sra/data_covid_19.tsv'

logging.basicConfig(filename=os.path.join(LOG_DIR, "processing_covid_19.log"), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_and_print(message, level=logging.INFO):
    """Log a message and print it to console."""
    print(message)
    logging.log(level, message)

def select_whitelist_file(technology, version):
    """Select the appropriate whitelist file based on technology and version."""
    whitelist_mapping = {
        "Single Cell 3' v1": "737K-april-2014_rc.txt",
        "Single Cell 3' v2": "737K-august-2016.txt",
        "Single Cell 3' v3": "3M-february-2018.txt",
        "Single Cell 3' v3.1": "3M-february-2018.txt",
        "Single Cell 3' HT v3.1": "3M-february-2018.txt",
        "Single Cell 3' v4": "3M-3pgex-may-2023.txt.gz",
        "Single Cell 5' v1": "737K-august-2016.txt",
        "Single Cell 5' v2": "737K-august-2016.txt",
        "Single Cell 5' HT v2": "737K-august-2016.txt",
        "Single Cell 5' v3": "3M-5pgex-jan-2023.txt.gz",
    }

    whitelist_file = whitelist_mapping.get(f"{technology} {version}", None)

    if whitelist_file is None:
        log_and_print(f"Could not find a matching whitelist file for technology {technology} and version {version}.", logging.ERROR)
        return None

    whitelist_file_path = os.path.join(WHITELIST_DIR, whitelist_file)

    if not os.path.exists(whitelist_file_path):
        log_and_print(f"Whitelist file {whitelist_file_path} does not exist.", logging.ERROR)
        return None

    log_and_print(f"Selected whitelist file: {whitelist_file_path}")
    return whitelist_file_path

def download_fastqs(run_accession):
    """Download FASTQ files using fasterq-dump."""
    fastq_dir = os.path.join(DOWNLOAD_DIR, run_accession)
    os.makedirs(fastq_dir, exist_ok=True)

    cmd = [
        FASTERQ_DUMP_PATH,
        "--outdir", fastq_dir,
        "--split-files",
        run_accession
    ]

    log_and_print(f"Downloading FASTQ files for {run_accession}...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_and_print(f"Successfully downloaded FASTQ files for {run_accession}.")
        return True
    except subprocess.CalledProcessError as e:
        log_and_print(f"Failed to download FASTQ files for {run_accession}: {e}", logging.ERROR)
        return False

def detect_barcode_length(fastq_files):
    """Detect the barcode length from the FASTQ files, return 0 if multiple lengths are detected."""
    barcode_lengths = set()
    for fastq_file in fastq_files:
        with open(fastq_file, 'r') as f:
            for _ in range(4):  # Read first 4 lines of the FASTQ file
                if _ == 1:  # The second line contains the sequence
                    barcode = next(f).strip()
                    barcode_lengths.add(len(barcode))
                else:
                    next(f)
        if len(barcode_lengths) > 1:
            return 0  # Multiple lengths detected

    if len(barcode_lengths) == 1:
        return barcode_lengths.pop()
    else:
        log_and_print(f"No barcode length detected.", logging.WARNING)
        return 0  # Default to 0 if no length detected

def select_fastq_files(fastq_files):
    """Select the appropriate fastq files for alignment."""
    fastq_files = sorted(fastq_files)
    r1_files = [f for f in fastq_files if '_1.fastq' in f]
    r2_files = [f for f in fastq_files if '_2.fastq' in f]
    r3_files = [f for f in fastq_files if '_3.fastq' in f]

    if r2_files and r3_files:
        log_and_print(f"Selected *2 and *3 FASTQ files for alignment.")
        return [r2_files[0], r3_files[0]]
    elif r1_files and r2_files:
        log_and_print(f"Selected *1 and *2 FASTQ files for alignment.")
        return [r1_files[0], r2_files[0]]
    elif r1_files:
        log_and_print(f"Only one FASTQ file available. Performing single-end alignment.")
        return [r1_files[0]]
    else:
        log_and_print(f"No suitable FASTQ files found for alignment.", logging.WARNING)
        return []

def run_star_alignment(fastq_files, output_prefix, run_accession, technology, version):
    """Run STAR alignment with specified parameters and handle errors gracefully."""
    barcode_length = detect_barcode_length(fastq_files)
    whitelist_file = select_whitelist_file(technology, version)

    if not whitelist_file or not os.path.exists(whitelist_file):
        log_and_print(f"Whitelist file {whitelist_file} does not exist.", logging.ERROR)
        return False

    log_file = os.path.join(LOG_DIR, f"{run_accession}_STAR.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    cmd = [
        'nice', '-n', '-20',
        STAR_PATH,
        "--runThreadN", "56",
        "--genomeDir", STAR_INDEX,
        "--outFileNamePrefix", os.path.join(MAPPED_DIR, run_accession, run_accession + "_"),
        "--soloType", "CB_UMI_Simple",
        "--soloFeatures", "Gene", "GeneFull",
        "--soloCBwhitelist", whitelist_file,
        "--soloBarcodeReadLength", str(barcode_length),
        "--runDirPerm", "All_RWX",
        "--readFilesIn"
    ] + fastq_files

    if len(fastq_files) == 1:
        cmd.extend(["--readFilesCommand", "zcat"])

    log_and_print(f"Running STAR alignment for {run_accession}...")
    log_and_print(f"STAR command: {' '.join(cmd)}")

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        log_and_print(f"STAR alignment completed successfully for {run_accession}.")
        return True
    except subprocess.CalledProcessError as e:
        log_and_print(f"Error during STAR alignment for {run_accession}: {e}", logging.ERROR)
        return False

def check_matrix_dimensions(run_accession):
    """Check and log the length of barcodes, features, and the shape of the matrix."""
    output_dir = os.path.join(MAPPED_DIR, run_accession, f"{run_accession}_Solo.out", "GeneFull", "raw")
    barcodes_file = os.path.join(output_dir, "barcodes.tsv")
    features_file = os.path.join(output_dir, "features.tsv")
    matrix_file = os.path.join(output_dir, "matrix.mtx")

    if not (os.path.exists(barcodes_file) and os.path.exists(features_file) and os.path.exists(matrix_file)):
        log_and_print(f"Required files are missing for {run_accession}. Skipping.")
        return None

    try:
        # Read barcodes, features, and matrix
        barcodes = pd.read_csv(barcodes_file, header=None)[0].tolist()
        features = pd.read_csv(features_file, header=None, sep='\t')
        feature_names = features[0].tolist()
        matrix = scipy.io.mmread(matrix_file).tocsr()

        num_barcodes = len(barcodes)
        num_features = len(feature_names)
        matrix_shape = matrix.shape

        log_and_print(f"{run_accession}: Number of barcodes: {num_barcodes}, Number of features: {num_features}, Matrix shape: {matrix_shape}")

        # Check if dimensions match
        if num_barcodes != matrix_shape[1] or num_features != matrix_shape[0]:
            log_and_print(f"Dimension mismatch in {run_accession}: Barcodes ({num_barcodes}) vs Matrix Columns ({matrix_shape[1]}), Features ({num_features}) vs Matrix Rows ({matrix_shape[0]})", logging.ERROR)
            return False

        return True
    except Exception as e:
        log_and_print(f"Error checking dimensions for {run_accession}: {e}", logging.ERROR)
        return False

def convert_star_to_h5ad(run_accession):
    """Convert STAR output files to H5AD format, handling dimensions properly."""
    output_dir = os.path.join(MAPPED_DIR, run_accession, f"{run_accession}_Solo.out", "GeneFull", "raw")
    barcodes_file = os.path.join(output_dir, "barcodes.tsv")
    features_file = os.path.join(output_dir, "features.tsv")
    matrix_file = os.path.join(output_dir, "matrix.mtx")
    h5ad_file = os.path.join(H5AD_DIR, f"{run_accession}.h5ad")

    if not all(os.path.exists(f) for f in [barcodes_file, features_file, matrix_file]):
        log_and_print(f"Required files are missing for {run_accession}. Skipping.", logging.ERROR)
        return False

    if not check_matrix_dimensions(run_accession):
        log_and_print(f"Skipping H5AD conversion for {run_accession} due to dimension mismatch.", logging.ERROR)
        return False

    try:
        log_and_print(f"Converting STAR output to H5AD for {run_accession}...")
        barcodes = pd.read_csv(barcodes_file, header=None)
        features = pd.read_csv(features_file, header=None, sep='\t', names=['gene_id', 'gene_name'])
        matrix = scipy.io.mmread(matrix_file).tocsr()

        adata = anndata.AnnData(X=matrix.T, obs=pd.DataFrame(index=barcodes), var=features.set_index('gene_id'))
        adata.write(h5ad_file)
        log_and_print(f"Successfully converted to H5AD: {h5ad_file}")

        # Deleting FASTQ files after successful conversion
        fastq_dir = os.path.join(DOWNLOAD_DIR, run_accession)
        if os.path.exists(fastq_dir):
            log_and_print(f"Deleting FASTQ files for {run_accession}...")
            for fastq_file in glob.glob(os.path.join(fastq_dir, '*.fastq')):
                os.remove(fastq_file)
            os.rmdir(fastq_dir)
            log_and_print(f"Successfully deleted FASTQ files for {run_accession}.")

        return True
    except Exception as e:
        log_and_print(f"Error converting STAR output to H5AD for {run_accession}: {e}", logging.ERROR)
        return False

def process_run(run_info):
    """Process each run by checking for existing files, downloading, aligning, and converting data."""
    run_accession, technology, version = run_info['Run'], run_info['Technology'], run_info['version']
    run_dir = os.path.join(MAPPED_DIR, run_accession)
    os.makedirs(run_dir, exist_ok=True)

    log_and_print(f"Starting processing for run: {run_accession}")

    # Check if the matrix files already exist
    if check_matrix_dimensions(run_accession) is not None:
        log_and_print(f"Matrix files found for {run_accession}. Proceeding to H5AD conversion.")
        if convert_star_to_h5ad(run_accession):
            return True
        else:
            log_and_print(f"Skipping {run_accession} due to conversion failure.", logging.ERROR)
            return False

    # Check if FASTQ files exist
    fastq_files = glob.glob(os.path.join(DOWNLOAD_DIR, run_accession, '*.fastq'))
    if not fastq_files:
        log_and_print(f"FASTQ files not found for {run_accession}. Initiating download.")
        if not download_fastqs(run_accession):
            log_and_print(f"Skipping {run_accession} due to FASTQ download failure.", logging.ERROR)
            return False
        fastq_files = glob.glob(os.path.join(DOWNLOAD_DIR, run_accession, '*.fastq'))

    # Select FASTQ files for alignment
    selected_fastq_files = select_fastq_files(fastq_files)
    if not selected_fastq_files:
        log_and_print(f"Skipping {run_accession} due to missing suitable FASTQ files.", logging.ERROR)
        return False

    # Run STAR alignment
    if not run_star_alignment(selected_fastq_files, run_dir, run_accession, technology, version):
        log_and_print(f"Skipping {run_accession} due to STAR alignment failure.", logging.ERROR)
        return False

    # Convert STAR output to H5AD format
    if not convert_star_to_h5ad(run_accession):
        log_and_print(f"Skipping {run_accession} due to conversion failure.", logging.ERROR)
        return False

    log_and_print(f"Completed processing for run: {run_accession}")
    return True


def main():
    metadata = pd.read_csv(METADATA_FILE, sep="\t")
    run_info_list = metadata[['Run', 'Technology', 'version']].to_dict(orient='records')

    # Parallel processing using multiprocessing Pool
    with Pool(processes=8) as pool:
        results = pool.map(process_run, run_info_list)

    log_and_print(f"Processing completed. Successful runs: {sum(results)}, Failed runs: {len(results) - sum(results)}")

if __name__ == "__main__":
    main()
