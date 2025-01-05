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
   surv = Survey()
   surv.add_log(data=log1, name="Position 1")
   surv.add_log(data=log2, name="Position 2")
   ```
4. **Analyse the Survey Data**\
   The following are methods of the Survey() object representing the typical use cases for acoustic consultants in the UK.
   ### Survey.resi_summary()\
   This method provides a summary of the measurement data for residential projects, with a focus on typical assessment procedures in the UK.
   It presents A-weighted Leqs for each day and night period (and evenings, if enabled), as well as the nth-highest LAmax during each night-time period.
   Optional arguments are:\
   **leq_cols** *List of tuples* *(default [("Leq", "A")]* Which column(s) you want to present as Leqs - this can be any Leq or statistical column.\
   **max_cols** *List of tuples* *(default [("Leq", "A")]* Which column(s) you want to present as an nth-highest value - this can be any column.\
   **lmax_n** *Int* *(default 10)* The nth-highest value to present.\
   **lmax_t** *Str* *(default "2min")* The time period T over which Lmaxes are presented. This must be equal to or longer than the period of the raw data.\
   \
   ### Survey.get_modal_l90()\
   \
   ### Survey.get_lmax_spectra()\
   Compute the Lmax Event spectra for the nth-highest Lmax during each night-time period.\
   **Note** the date presented alongside the Lmax event is actually the starting date of the night-time period. i.e. an Lmax event with a stamp of 20/12/2024 at 01:22 would actually have occurred on 21/12/2024 at 01:22. These stamps can also sometimes be out by a minute (known bug).
   \
   ### Survey.get_typical_leq_spectra()\
   Compute the Leq spectra for daytime, evening (if enabled) and night-time periods. This will present the overall Leqs across the survey, not the Leq for each day.
   \

### Other methods
The following are methods of the Survey() object which may also be of use\
### Known issues
- Lmax night-time timestamps can sometimes by out by a minute.\
## Troubleshooting
### ValueError: NaTType does not support time
This error occurs when the source csv file contains empty cells. It usually occurs when you have entered data into some row(s) or column(s) and then deleted it, leaving previously-full cells which are now empty.\
**Solution:** Create a new tab in your source csv file, and paste in your headers and data as you wish it to be presented to the toolkit, avoiding having to delete any columns and rows. Delete the old tab. If you do have to delete any data in the new tab, you will need to repeat the process to ensure this error is not thrown up again.

### Terms of use
The PyNS toolkit was built by Tony Trup of [Timbral(https://www.timbral.co.uk)].
I accept no liability for the outputs of this toolkit. You use it at your own risk, and you should carry out your own checks and balances to ensure you are satistfied that the output is accurate.
This is an open source project, and I welcome suggestions for changes, improvements or new features. You can also write your own methods or functions and share them with me, either by getting in touch offline, or by creating a new branch from this Git repository.
You may use this toolkit subject to the licence conditions below. For clarity, you may use this toolkit or adaptations of it in your day-to-day engineering work, but incorporating it into a commercial software product or service is not permitted.
This project is being shared under a [Creative Commons CC BY-NC-SA 4.0 Licence (https://creativecommons.org/licenses/by-nc-sa/4.0/)].
Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
NonCommercial — You may not use the material for commercial purposes.
ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
