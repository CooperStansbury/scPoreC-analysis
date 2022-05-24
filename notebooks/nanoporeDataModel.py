"""
This set fo data models comes from https://github.com/nanoporetech/pore-c/tree/a865b7ea64d7a972fe878e8f9ea4aa7553def363
"""

import numpy as np
import pandas as pd
import dask

import pysam
from pysam import FastaFile, FastxFile
from pysam import AlignmentFile
from Bio import Restriction
from Bio.Seq import Seq

import pyranges as pr
from enum import Enum
from typing import Dict, List, NewType, Optional
from intake.source.base import DataSource, Schema
import pyarrow as pa
from pydantic import BaseModel, confloat, conint, constr


GENOMIC_COORD_DTYPE = "uint32"  # should be fine as long as individual chromosomes are less than 4Gb
GENOMIC_DISTANCE_DTYPE = "int32"
READ_COORD_DTYPE = "uint32"
READ_DISTANCE_DTYPE = "int32"
STRAND_DTYPE = "bool"
FRAG_IDX_DTYPE = "uint32"
READ_IDX_DTYPE = "uint32"
ALIGN_IDX_DTYPE = "uint32"
PERCENTAGE_DTYPE = "float32"
HAPLOTYPE_IDX_DTYPE = "int8"
MQ_DTYPE = "uint8"
ALIGN_SCORE_DTYPE = "uint32"
PHASE_SET_DTYPE = GENOMIC_COORD_DTYPE
HAPLOTYPE_IDX_DTYPE = "int8"
PERCENTAGE_DTYPE = "float32"

INPUT_REFGENOME_REGEX = r"(.+)\.(fasta|fa|fna)(\.gz)*"

PQ_ENGINE = "pyarrow"
PQ_VERSION = "2.0"

SHORT_RANGE_CUTOFF = 20_000


PHRED_TO_PROB = np.power(10, (np.arange(256, dtype=float) / -10.0))

def mean_qscore(quals):
    return -10 * np.log10(PHRED_TO_PROB[quals].mean())


class IndexedFasta(DataSource):
    name = "indexed_bedfile"
    version = "0.1.0"
    container = "python"
    partition_access = True
    description = "A bgzipped and indexed fasta file"

    def __init__(self, urlpath, metadata=None):
        self._urlpath = urlpath
        self._dataset = None
        self._dtype = None
        self._chroms = None
        super(IndexedFasta, self).__init__(metadata=metadata)

    def _open_dataset(self):
        self._dataset = FastaFile(self._urlpath)

    def _get_schema(self):
        if self._dataset is None:
            self._open_dataset()
        self._chroms = list(self._dataset.references)
        chrom_lengths = [{"chrom": t[0], "length": t[1]} for t in zip(self._dataset.references, self._dataset.lengths)]
        return Schema(
            datashape=None,
            dtype=None,
            shape=None,
            npartitions=len(self._chroms),
            extra_metadata={"chroms": chrom_lengths},
        )

    def _get_partition(self, i):
        chrom = self._chroms[i]
        return [{"seqid": chrom, "seq": self._dataset.fetch(chrom)}]

    def read_chunked(self):
        self._load_metadata()
        for i in range(self.npartitions):
            yield self._get_partition(i)

    def to_dask(self):
        from dask import bag as db

        self._load_metadata()
        return db.from_delayed([dask.delayed(self._get_partition(i)) for i in range(self.npartitions)])

    def _close(self):
        # close any files, sockets, etc
        if self._dataset is not None:
            self._dataset.close()
            
            
            
            
class _BaseModel(BaseModel):
    @classmethod
    def pandas_dtype(cls, overrides=None):
        res = {}
        if overrides is None:
            overrides = {}
        overrides = overrides if overrides is not None else {}
        schema = cls.schema()
        for column, col_schema in schema["properties"].items():
            if column in overrides:
                res[column] = overrides[column]
                continue
            if "$ref" in col_schema:
                def_key = col_schema["$ref"].rsplit("/", 1)[-1]
                col_schema = schema["definitions"][def_key]
            if "dtype" in col_schema:
                dtype = col_schema["dtype"]
            else:
                assert "enum" in col_schema
                dtype = pd.CategoricalDtype(col_schema["enum"], ordered=True)
            res[column] = dtype
        return res

    @classmethod
    def pyarrow_schema(cls, overrides=None):
        dtype = cls.pandas_dtype(overrides=overrides)
        df = pd.DataFrame(columns=dtype.keys(), dtype=dtype)
        return pa.Schema.from_pandas(df)

    def to_tuple(self):
        return tuple([_[1] for _ in self])

    @classmethod
    def to_dataframe(cls, data: List, overrides=Optional[Dict]):
        columns = [a[0] for a in data[0]]
        dtype = cls.pandas_dtype(overrides=overrides)
        df = pd.DataFrame([a.to_tuple() for a in data], columns=columns).astype(dtype)
        return df


class AlignmentType(str, Enum):
    unmapped = "unmapped"
    primary = "primary"
    secondary = "secondary"
    supplementary = "supplementary"


class FragmentRecord(_BaseModel):
    """Meta-data associated with a restriction fragments"""

    chrom: constr(min_length=1, strip_whitespace=True)
    start: conint(ge=0)
    end: conint(ge=0)
    fragment_id: conint(ge=1, strict=True)
    fragment_length: conint(ge=1, strict=True)

    class Config:
        use_enum_values = True
        fields = dict(
            chrom=dict(description="The chromosome/contig the fragment is derived from", dtype="category"),
            start=dict(
                description="The zero-based start position on the genome of the fragment", dtype=GENOMIC_COORD_DTYPE
            ),
            end=dict(
                description="The zero-based end position on the genome of the fragment", dtype=GENOMIC_COORD_DTYPE
            ),
            fragment_id=dict(description="Unique integer ID of the fragment, starts at 1", dtype=FRAG_IDX_DTYPE),
            fragment_length=dict(description="Length of the fragment", dtype=GENOMIC_COORD_DTYPE),
        )


class AlignmentRecord(_BaseModel):
    """A subset of the fields in the BAM file"""

    read_idx: conint(ge=0, strict=True)
    align_idx: conint(ge=0, strict=True)
    align_type: AlignmentType
    chrom: constr(min_length=1, strip_whitespace=True)
    start: conint(ge=0)
    end: conint(ge=0)
    strand: STRAND_DTYPE
    read_name: constr(min_length=1, strip_whitespace=True)
    read_length: conint(ge=1)
    read_start: conint(ge=0)
    read_end: conint(ge=0)
    mapping_quality: conint(ge=0, le=255)
    align_score: conint(ge=0)
    align_base_qscore: conint(ge=0)
    phase_set: int = 0
    phase_qual: conint(ge=0) = 0
    haplotype: conint(ge=-1) = -1

    class Config:
        use_enum_values = True
        fields = dict(
            read_idx=dict(description="Unique integer ID of the read", dtype=READ_IDX_DTYPE),
            align_idx=dict(description="Unique integer ID of the aligned segment", dtype=ALIGN_IDX_DTYPE),
            align_type=dict(description="The type of alignment", dtype="category"),
            chrom=dict(description="The chromosome/contig the read is aligned to", dtype="category"),
            start=dict(
                description="The zero-based start position on the genome of the alignment", dtype=GENOMIC_COORD_DTYPE
            ),
            end=dict(description="The end position on the genome of the alignment", dtype=GENOMIC_COORD_DTYPE),
            strand=dict(description="The alignment strand", dtype="bool"),
            read_name=dict(description="The original read name", dtype="str"),
            read_length=dict(description="The length of the read in bases", dtype=READ_COORD_DTYPE),
            read_start=dict(description="The start coordinate on the read (0-based)", dtype=READ_COORD_DTYPE),
            read_end=dict(description="The end coordinate on the read (0-based)", dtype=READ_COORD_DTYPE),
            mapping_quality=dict(description="The mapping quality as calculated by the aligner", dtype=MQ_DTYPE),
            align_score=dict(description="The alignment score as calculated by the aligner", dtype=ALIGN_SCORE_DTYPE),
            align_base_qscore=dict(
                description="The mean read base score for the aligned segment (rounded to the nearest integer).",
                dtype=ALIGN_SCORE_DTYPE,
            ),
            phase_set=dict(
                description="The ID of the phase set, often this is the start position of the phase block",
                dtype=PHASE_SET_DTYPE,
            ),
            phase_qual=dict(description="The phred-scaled quality score of the haplotype assignment", dtype=MQ_DTYPE),
            haplotype=dict(
                description=(
                    "The id of the haplotype within this block, usually set to 1 or 2. "
                    "A value of -1 means that this alignment is unphased"
                ),
                dtype=HAPLOTYPE_IDX_DTYPE,
            ),
        )

    @classmethod
    def from_aligned_segment(cls, align: pysam.AlignedSegment) -> "AlignmentRecord":
        """Extract information from a pysam Aligned segment"""
        read_name, read_idx, align_idx = align.query_name.split(":")
        read_idx, align_idx = int(read_idx), int(align_idx)

        if align.is_unmapped:
            align_cat = "unmapped"
            chrom, start, end, align_score = "NULL", 0, 0, 0
            read_length = align.query_length
            quals = align.query_qualities
            # TODO: handle this more gracefully
            if quals is None:
                align_base_qscore = 0
            else:
                align_base_qscore = mean_qscore(np.array(align.query_qualities))
        else:
            chrom, start, end = (align.reference_name, align.reference_start, align.reference_end)
            read_length = align.infer_read_length()
            align_score = align.get_tag("AS")
            align_base_qscore = mean_qscore(np.array(align.query_alignment_qualities))
            if align.is_secondary:
                align_cat = "secondary"
            elif align.is_supplementary:
                align_cat = "supplementary"
            else:
                align_cat = "primary"

        optional = {}
        for key, tag in [("haplotype", "HP"), ("phase_set", "PS"), ("phase_qual", "PC")]:
            if align.has_tag(tag):
                optional[key] = int(align.get_tag(tag))
        return cls(
            read_idx=read_idx,
            align_idx=align_idx,
            align_type=align_cat,
            chrom=chrom,
            start=start,
            end=end,
            strand=not align.is_reverse,
            read_name=read_name,
            read_length=read_length,
            read_start=align.query_alignment_start,
            read_end=align.query_alignment_end,
            mapping_quality=align.mapq,
            align_score=align_score,
            align_base_qscore=np.rint(align_base_qscore),
            **optional,
        )

    @classmethod
    def to_dataframe(cls, aligns: List, chrom_order: List[str] = None):
        columns = [a[0] for a in aligns[0]]
        if chrom_order:
            overrides = {"chrom": pd.CategoricalDtype(chrom_order, ordered=True)}
        else:
            overrides = {}
        dtype = cls.pandas_dtype(overrides=overrides)
        df = pd.DataFrame([a.to_tuple() for a in aligns], columns=columns)
        df = df.astype(dtype)
        return df

    @staticmethod
    def update_dataframe_with_haplotypes(align_df, haplotype_df):
        if len(haplotype_df) == 0:
            logger.info(f"Aligment haplotypes dataframe is empty, haplotypes won't be added.")
            return align_df
        haplotype_df = (
            haplotype_df.join(
                haplotype_df["#readname"]
                .str.split(":", expand=True)
                .rename({0: "read_name", 1: "read_idx", 2: "align_idx"}, axis=1)
            )
            .rename(columns={"phaseset": "phase_set"})
            .replace(dict(haplotype={"none": -1, "H1": 1, "H2": 2}, phase_set={"none": 0},))
            .astype(
                {
                    "read_idx": READ_IDX_DTYPE,
                    "align_idx": ALIGN_IDX_DTYPE,
                    "haplotype": HAPLOTYPE_IDX_DTYPE,
                    "phase_set": PHASE_SET_DTYPE,
                }
            )
            .set_index(["read_idx", "align_idx"])
        )
        col_order = list(align_df.columns)
        align_df = align_df.set_index(["read_idx", "align_idx"])
        align_df["haplotype"] = -1
        align_df["phase_set"] = 0
        align_df["phase_qual"] = 0
        align_df.update(haplotype_df[["haplotype", "phase_set"]], overwrite=True, errors="ignore")
        align_df = align_df.reset_index()[col_order].astype(
            {"haplotype": HAPLOTYPE_IDX_DTYPE, "phase_set": PHASE_SET_DTYPE}
        )
        return align_df


class AlignmentFilterReason(str, Enum):
    null = "null"
    Pass = "pass"
    unmapped = "unmapped"
    singleton = "singleton"
    low_mq = "low_mq"
    short_overlap = "short_overlap"
    overlap_on_read = "overlap_on_read"
    not_on_shortest_path = "not_on_shortest_path"


class PoreCRecord(AlignmentRecord):
    pass_filter: bool = True
    filter_reason: AlignmentFilterReason = AlignmentFilterReason.null
    fragment_id: conint(ge=0) = 0
    num_contained_fragments: conint(ge=0) = 0
    num_overlapping_fragments: conint(ge=0) = 0
    overlap_length: conint(ge=0) = 0
    fragment_start: conint(ge=0) = 0
    fragment_end: conint(ge=0) = 0
    perc_of_alignment: confloat(ge=0, le=100) = 0.0
    perc_of_fragment: confloat(ge=0, le=100) = 0.0
    is_contained: bool = False

    class Config:
        use_enum_values = True
        fields = dict(
            pass_filter=dict(description="Boolean flag, true if alignment passes all filters", dtype="bool"),
            filter_reason=dict(
                description="If an alignment fails the filter the reason will be listed here", dtype="str" # changed by CS
            ),
            fragment_id=dict(
                description="The UID of the restriction fragment assigned to this alignment", dtype=FRAG_IDX_DTYPE
            ),
            num_contained_fragments=dict(
                description="The number of restriction fragments completely contained within this alignment",
                dtype="uint32",
            ),
            num_overlapping_fragments=dict(
                description="The number of restriction fragments overlapping this alignment", dtype="uint32"
            ),
            overlap_length=dict(
                description="The length of the overlap between alignment and fragment", dtype=GENOMIC_COORD_DTYPE
            ),
            fragment_start=dict(
                description="The start point on the genome of this restriction fragment", dtype=GENOMIC_COORD_DTYPE
            ),
            fragment_end=dict(
                description="The end point on the genome of this restriction fragment", dtype=GENOMIC_COORD_DTYPE
            ),
            perc_of_alignment=dict(
                description="The percentage of the aligned segment that overlaps the assigned fragment",
                dtype=PERCENTAGE_DTYPE,
            ),
            perc_of_fragment=dict(
                description="The percentage of the assigned restriction fragment that overlaps the aligned segment",
                dtype=PERCENTAGE_DTYPE,
            ),
            is_contained=dict(
                description="Boolean flag to inidicate if the alignment is fully contained with the fragment",
                dtype="bool",
            ),
        )

    @classmethod
    def init_dataframe(cls, align_df: "AlignmentRecordDf") -> "PoreCRecordDf":
        res = align_df.copy()
        schema = cls.schema()

        col_schema = schema["properties"]

        for key, val in col_schema.items():
            if "$ref" in val:
                def_key = val["$ref"].rsplit("/", 1)[-1]
                col_schema[key] = schema["definitions"][def_key]

        dtype = cls.pandas_dtype()
        additional_fields = set(dtype.keys()) - (set(align_df.index.names) | set(align_df.columns))
        num_rows = len(res)
        for column in [c for c in col_schema if c in additional_fields]:
            cs = col_schema[column]
            if "default" in cs:
                default_value = cs["default"]
            elif "enum" in cs:
                default_value = cs["enum"][0]
            else:
                raise ValueError(cs)
            res[column] = pd.Series([default_value] * num_rows, index=res.index).astype(dtype[column])
        return res

    
AlignmentRecordDf = NewType("AlignmentRecordDf", pd.DataFrame)
FragmentRecordDf = NewType("FragmentRecordDf", pd.DataFrame)
PoreCRecordDf = NewType("PoreCRecordDf", pd.DataFrame)
PoreCContactRecordDf = NewType("PoreCContactRecordDf", pd.DataFrame)
PoreCConcatemerRecordDf = NewType("PoreCConcatemerRecordDf", pd.DataFrame)

Chrom = NewType("Chrom", str)