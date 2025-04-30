#!/bin/bash

# Install pandoc if not already installed
if ! command -v pandoc &> /dev/null; then
    echo "Installing pandoc..."
    sudo apt-get update
    sudo apt-get install -y pandoc
fi

# Convert README.md to Word document
pandoc README.md -o Wine_Quality_Prediction_Model.docx

echo "Word document has been created: Wine_Quality_Prediction_Model.docx" 