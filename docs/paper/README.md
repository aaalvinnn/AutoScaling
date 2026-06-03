# AutoLFD Markdown Conversion Package

This package contains a Markdown conversion of the uploaded PDF paper and extracted visual assets.

## Contents

- `AutoLFD.md`: main Markdown version with local image embeds.
- `assets/figures/`: page-cropped figure images used by the Markdown file.
- `assets/source_images/`: original raster images extracted from the PDF object stream.
- `AutoLFD_raw_extracted_text.txt`: raw text extracted from the PDF for traceability.
- `source/tcc.pdf`: original PDF source file.
- `asset_manifest.json`: machine-readable list of included files.

## Notes

The PDF is a two-column academic paper. Figures 1-4 were available as raster objects, while several plots were vector-rendered in the PDF; those plots were extracted by cropping the rendered page regions so they can be embedded in Markdown.
