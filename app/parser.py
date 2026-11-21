"""Genomic parser: reads FASTA with BioPython and yields encoded chunks."""
from Bio import SeqIO
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .sequence_utils import encode


def iter_chunks(fasta_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Yield (record_id, offset, encoded_array). Chunks overlap by
    SITE_LEN-1 bases so every possible site lies in exactly one chunk."""
    step = chunk_size - overlap
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(rec.seq)
        for start in range(0, len(seq), step):
            piece = seq[start:start + chunk_size]
            if len(piece) > overlap:
                yield rec.id, start, encode(piece)
            if start + chunk_size >= len(seq):
                break


def genome_stats(fasta_path):
    """Return (number_of_records, total_bases)."""
    n, total = 0, 0
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        n += 1
        total += len(rec.seq)
    return n, total
