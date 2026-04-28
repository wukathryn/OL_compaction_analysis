"""Dataframe filters and pivots used before plotting and statistics.

Includes helpers to identify cells flagged for omission, drop cells with
incomplete time series, and restructure long-format analysis dataframes
into the wide formats expected by trajectory plotting and the repeated-
measures statistics in :mod:`run_stats`.
"""

import re
import pandas as pd


def list_cells_to_omit(df, omit_col, cell_id='UID'):
    """
    Returns a list of cells marked for omission in a given omit_col.

    Args:
        df (pd.DataFrame): Input dataframe.
        omit_col (str): Column that marks cells to omit (expects 'Y' for omission).
        cell_id (str): Column representing unique cell IDs (default 'UID').

    Returns:
        list: List of cell IDs to omit. Empty list if omit_col is not found.
    """
    if omit_col in df.columns:
        cells_to_omit = df.loc[df[omit_col] == 'Y', cell_id].unique().tolist()
        print(f'Cells omitted: {cells_to_omit}')
        return cells_to_omit
    else:
        print(f'{omit_col} is not a column in this dataframe.')
        return []


def filter_incomplete_data(df, time_col, cell_id='UID', value_col=None, max_num_incomplete=2):
    """
    Select timepoints shared across most cells (allowing flexibility
    for `max_num_incomplete` cells to have missing data at these timepoints)
    Then remove cells that are missing any data at the selected timepoints.

    Args:
        df (pd.DataFrame): Input dataframe.
        cell_id (str): Column name for cell ID.
        time_col (str): Column name for timepoint.
        value_col (str, optional): Column to check for non-NA data.
                                   If None, checks only for row presence.
        allowed_missing_cells (int): Maximum number of cells allowed to be missing at a timepoint.

    Returns:
        pd.DataFrame: Filtered dataframe with cells tha have data across all shared timepoints.
    """

    if value_col is not None:
        data_present = df[df[value_col].notna()]
    else:
        data_present = df

    n_total_cells = df[cell_id].nunique()

    # For each timepoint, which cells have data
    timepoint_cell_map = data_present.groupby(time_col)[cell_id].unique()

    valid_timepoints = []
    missing_cells = []

    for t, cells_with_data in timepoint_cell_map.items():
        n_cells_with_data = len(cells_with_data)
        if n_cells_with_data >= (n_total_cells - max_num_incomplete):
            valid_timepoints.append(t)
            # Track which cells are missing at this timepoint
            missing_at_t = set(df[cell_id].unique()) - set(cells_with_data)
            missing_cells.extend(missing_at_t)

    print(f'Valid timepoints kept: {valid_timepoints}')

    # Filter to valid timepoints
    df_valid_times = df[df[time_col].isin(valid_timepoints)].copy()

    # Cells missing at any valid timepoint will be excluded
    excluded_cells = set(missing_cells)
    print(f'Cells excluded for missing data: {list(excluded_cells)}')

    filtered_df = df_valid_times[~df_valid_times[cell_id].isin(excluded_cells)].copy()

    return filtered_df


def select_timepoints(df, time_col, init_tp=None, final_tp=None):
    filtered_df = df.copy()

    if init_tp is not None:
        filtered_df = filtered_df[filtered_df[time_col] >= init_tp]

    if final_tp is not None:
        filtered_df = filtered_df[filtered_df[time_col] <= final_tp]

    print(f'Timeponts kept: {filtered_df[time_col].unique().tolist()}')

    return filtered_df


def compute_change_cols(df, ycols, cell_id='UID', time_col='t'):
    """
    Compute change in and cumulative change in columns for multiple columns at once, optimized.
    Automatically sorts by [cell_id, time_col] before processing.

    Args:
        df (pd.DataFrame): Input dataframe.
        ycols (list of str): List of columns to compute changes for.
        cell_id (str): Column representing unique cell IDs.
        time_col (str): Column representing time ordering.

    Returns:
        pd.DataFrame, list: Updated dataframe and updated list of ycols including new columns.
    """
    df = df.sort_values([cell_id, time_col]).copy()

    uid_changes = df[cell_id] != df[cell_id].shift()

    # Vectorized diff across multiple columns
    diffs = df[ycols].diff()
    diffs.loc[uid_changes, :] = 0
    diffs.columns = [f'change in {col}' for col in ycols]

    cumsums = diffs.cumsum()
    cumsums = cumsums.subtract(cumsums.where(uid_changes).ffill().fillna(0))
    cumsums.columns = [f'cumulative change in in {col}' for col in ycols]

    df = pd.concat([df, diffs, cumsums], axis=1)
    updated_ycols = ycols + list(diffs.columns) + list(cumsums.columns)
    return df, updated_ycols


def compute_means_by_biorep(df, groupbycols, ycols, omit_col='omit', cell_id='UID'):
    """
    Compute means for biological replicates.

    Args:
        df (pd.DataFrame): Input dataframe.
        groupbycols (list of str): Columns to group by.
        ycols (list of str): Columns to compute means for.
        cell_id (str): Column representing unique cell IDs.
        omit_col (str): Column that marks cells to omit (expects 'Y' for omission).

    Returns:
        pd.DataFrame: Grouped dataframe with means and number of cells averaged.
    """

    if omit_col is not None:
        to_omit = list_cells_to_omit(df, omit_col, cell_id=cell_id)
        df = df.loc[~df[cell_id].isin(to_omit)].copy()

    agg_cols = {ycol: 'mean' for ycol in ycols}

    # Group for means
    df_means = df.groupby(groupbycols, as_index=False).agg(agg_cols)

    # Group for counts (count number of unique cells per group)
    df_counts = (
        df.groupby(groupbycols)[cell_id]
        .nunique()
        .reset_index()
        .rename(columns={cell_id: 'num cells averaged'})
    )

    # Merge means and counts
    df_biorep = pd.merge(df_means, df_counts, on=groupbycols)

    return df_biorep
