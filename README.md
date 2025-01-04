# PyNS - Toolkit for Analysing Noise Survey Data
## Overview
PyNS is a Python-based toolkit designed to assist acoustic consultants and engineers in analyzing noise survey data. The library provides various tools and utilities for processing, interpreting, and visualizing noise measurements.

## Features
- Import and process noise survey data from .csv files
- Analyse and recompute sound levels and statistical meausres (Leq, Lmax, L90 etc.)
- Handle large datasets from multiple measurement positions efficiently
- Visualise noise data through customisable plots (WIP)
- Generate reports with summary statistics (WIP)

## Installation
The library is not currently available through pip or Anaconda (we're working on it). You can pull the code from git, or simply download the files on this GitHub page.

## Usage
### Basic Workflow
1. **Import the library**\
   Import the library into your script or active console.
   ```
   from PyNS import*
   ```
2. **Load Data**\
   A single measurement position can be loaded into a Log object.
   ```
   log1 = Log(path="path/to/data_for_pos1.csv")
   log2 = Log(path="path/to/data_for_pos2.csv")
   ```
3. **Combine Data**\
   The data from multiple measurement positions can be combined together within a Survey object.
   First we need to create a Survey object, and then add each Log one at a time.
   ```
   survey = Survey()
   survey.add_log(data=log1, name="Position 1")
   survey.add_log(data=log2, name="Position 2")
   ```
4. **Analyse the Survey Data**\
   ### Resi Summary\
   The survey.resi_summary() method provides a summary of the measurement data for residential projects, with a focus on typical assessment procedures in the UK.
   It presents A-weighted Leqs for each day and night period (and evenings, if enabled), as well as the nth-highest LAmax during each night-time period.
   Optional arguments are:\
   **leq_cols** *List of tuples* *(default [("Leq", "A")]* Which column(s) you want to present as Leqs - this can be any Leq or statistical column.
   **max_cols** *(default A)* Which column(s) you want to present as an nth-highest value - this can be any column.
   **lmax_n** (default 10)*

### Terms of use
This is an open source project, and I am open to suggestions for changes, improvements or new features.
You may use this toolkit subject to the licence conditions below. For clarity, you may use this toolkit or adaptations of it in your day-to-day engineering work, but incorporating it into a commercial software product or service is not permitted.
This project is being shared under a [Creative Commons CC BY-NC-SA 4.0 Licence (https://creativecommons.org/licenses/by-nc-sa/4.0/)].
Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
NonCommercial — You may not use the material for commercial purposes.
ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
