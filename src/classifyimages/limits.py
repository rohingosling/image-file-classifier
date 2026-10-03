#-----------------------------------------------------------------------------------------------------------------------
# Module:  limits.py
# Project: classifyimages
# Version: 1.0.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Keep tested collection and inference bounds available without importing image or model libraries.
#-----------------------------------------------------------------------------------------------------------------------

# Tested collection limit and bounded CPU inference batch size.

MAX_COLLECTION_IMAGES = 2_000
INFERENCE_BATCH_SIZE  = 8
