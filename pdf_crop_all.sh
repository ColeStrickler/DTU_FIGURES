#!/bin/bash


find ./figures/ -name "*.pdf" -type f -exec sh -c '
for f do
    pdfcrop "$f" "${f%.pdf}-tmp.pdf" && mv "${f%.pdf}-tmp.pdf" "$f"
done
' sh {} +