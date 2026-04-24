API
===

Import cnvturbo together with scanpy as

.. code-block:: python

   import scanpy as sc
   import cnvturbo as cnv

For consistency, the cnvturbo API tries to follow the `scanpy API <https://scanpy.readthedocs.io/en/stable/api/index.html>`__
as closely as possible.

.. _api-io:

Input/Output: `io`
------------------

.. module:: cnvturbo.io

.. autosummary::
   :toctree: ./generated

   genomic_position_from_biomart
   genomic_position_from_gtf
   read_scevan


Preprocessing: `pp`
-------------------

.. module:: cnvturbo.pp

.. autosummary::
   :toctree: ./generated

   neighbors


Tools: `tl`
-----------

Tools add an interpretable annotation to the :class:`~anndata.AnnData` object
which usually can be visualized by a corresponding plotting function.

The tools for embeddings and clustering mirror the scanpy API.
However, while the scanpy tools operate on transcriptomics data, the
cnvturbo equivalent operates on CNV data.

.. module:: cnvturbo.tl


InferCNV
^^^^^^^^

.. autosummary::
   :toctree: ./generated

   infercnv
   copykat

CNV scores
^^^^^^^^^^

.. autosummary::
   :toctree: ./generated

   cnv_score
   ithcna
   ithgex

Embeddings
^^^^^^^^^^

.. autosummary::
   :toctree: ./generated

   pca
   umap
   tsne

Clustering
^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   leiden



Plotting: `pl`
--------------

.. module:: cnvturbo.pl

InferCNV
^^^^^^^^

.. autosummary::
   :toctree: ./generated

   chromosome_heatmap
   chromosome_heatmap_summary

Embeddings
^^^^^^^^^^
.. autosummary::
   :toctree: ./generated

   umap
   tsne


Datasets: `datasets`
--------------------

.. module:: cnvturbo.datasets

.. autosummary::
   :toctree: ./generated

   maynard2020_3k
   oligodendroglioma
