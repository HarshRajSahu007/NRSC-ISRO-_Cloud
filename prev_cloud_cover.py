"""
Cloud Detection System - Python Version
Based on the MATLAB implementation for INSAT-3D cloud detection algorithms.
Completely revised with proper dimension handling and error fixes.
"""

import os
import glob
import numpy as np
import h5py
import datetime
from scipy import stats
import netCDF4 as nc
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import calendar
import imageio
from scipy.io import loadmat, savemat
import logging

def robust_loadmat(filename):
    """Handle both v7.3 and older .mat files with proper array orientation"""
    try:
        data = loadmat(filename)
        # Check if any arrays need transposing
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 2:
                # Check if array appears transposed (common in MATLAB to Python conversion)
                if key in ['ms', 'topo_N', 'ln', 'lt']:  # Known arrays that might need transposing
                    if data[key].shape[0] == 1 or data[key].shape[1] == 1:
                        continue  # Don't transpose vectors
                    if data[key].shape[0] < data[key].shape[1]:
                        data[key] = data[key].T
        return data
    except NotImplementedError:
        # Handle v7.3 HDF5 format
        try:
            data = {}
            with h5py.File(filename, 'r') as f:
                for key in f.keys():
                    # Convert to numpy array and handle orientation
                    arr = np.array(f[key])
                    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                        arr = arr.T
                    data[key] = arr
            return data
        except Exception as e:
            print(f"Error loading MAT file {filename}: {str(e)}")
            return None

def standardize_dims(*arrays):
    """Ensure all arrays have the same dimensions with better error handling
    
    Args:
        *arrays: Variable number of numpy arrays to standardize
        
    Returns:
        Standardized arrays with matching dimensions (center cropped or zero-padded)
        Returns single array if only one input, tuple otherwise
    """
    if not arrays:
        return tuple()
    
    # Filter out None values and validate inputs
    valid_arrays = []
    for arr in arrays:
        if arr is None:
            valid_arrays.append(None)
            continue
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(arr)}")
        if arr.ndim < 2:
            raise ValueError("Input arrays must be at least 2D")
        valid_arrays.append(arr)
    
    if not valid_arrays:
        return arrays[0] if len(arrays) == 1 else arrays
    
    # Check if all arrays already have the same shape
    ref_shape = valid_arrays[0].shape
    if all(arr.shape == ref_shape for arr in valid_arrays if arr is not None):
        return arrays[0] if len(arrays) == 1 else arrays
    
    # Find minimum dimensions across all arrays
    min_dims = []
    for dim in range(len(ref_shape)):
        min_dim = min(arr.shape[dim] for arr in valid_arrays if arr is not None)
        min_dims.append(min_dim)
    target_shape = tuple(min_dims)
    
    standardized = []
    for arr in arrays:
        if arr is None:
            standardized.append(None)
            continue
            
        if arr.shape == target_shape:
            standardized.append(arr)
            continue
            
        try:
            # Calculate padding or cropping needed for each dimension
            pads = []
            crops = []
            for dim in range(arr.ndim):
                diff = arr.shape[dim] - target_shape[dim]
                if diff > 0:
                    # Need to crop
                    crop_before = diff // 2
                    crop_after = diff - crop_before
                    crops.append((crop_before, crop_after))
                else:
                    crops.append((0, 0))
                    
            # Perform cropping
            if any(crops):
                slices = []
                for crop in crops:
                    start = crop[0]
                    end = arr.shape[len(slices)] - crop[1] if crop[1] > 0 else None
                    slices.append(slice(start, end))
                cropped = arr[tuple(slices)]
            else:
                cropped = arr
                
            standardized.append(cropped)
            
        except Exception as e:
            print(f"Error standardizing array from {arr.shape} to {target_shape}: {str(e)}")
            # Fallback to zero-padding
            try:
                pad_width = []
                for dim in range(arr.ndim):
                    diff = target_shape[dim] - arr.shape[dim]
                    if diff > 0:
                        pad_before = diff // 2
                        pad_after = diff - pad_before
                        pad_width.append((pad_before, pad_after))
                    else:
                        pad_width.append((0, 0))
                
                padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
                standardized.append(padded)
            except Exception as e:
                print(f"Failed to pad array: {str(e)}")
                standardized.append(np.zeros(target_shape))
    
    # Return single array if only one input, otherwise return tuple
    if len(standardized) == 1:
        return standardized[0]
    return tuple(standardized)
    
def validate_h5_file(filename):
    """Validate HDF5 file structure and contents"""
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False
            
        with h5py.File(filename, 'r') as f:
            print("\nFile structure validation:")
            available_keys = list(f.keys())
            print("Available datasets:", available_keys)
            
            required_datasets = [
                'IMG_VIS', 'IMG_VIS_RADIANCE',
                'IMG_MIR', 'IMG_MIR_TEMP',
                'IMG_TIR1', 'IMG_TIR1_TEMP',
                'IMG_TIR2', 'IMG_TIR2_TEMP',
                'IMG_WV', 'IMG_WV_TEMP',
                'Sat_Elevation', 'Sun_Elevation'
            ]
            
            for ds in required_datasets:
                if ds not in available_keys:
                    print(f"Warning: Required dataset {ds} not found")
                else:
                    data = f[ds]
                    print(f"{ds}: shape={data.shape}, dtype={data.dtype}")
                    
            return True
    except Exception as e:
        print(f"Error validating HDF5 file: {str(e)}")
        return False

def read_h5_data(filename):
    """Robust HDF5 data reader that handles 3D INSAT-3D data"""
    data = {}
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None
            
        with h5py.File(filename, 'r') as f:
            # Read all required datasets with consistent naming convention
            datasets = {
                'count_VIS': 'IMG_VIS',
                'rad_VIS': 'IMG_VIS_RADIANCE',
                'count_MIR': 'IMG_MIR',
                'temp_MIR': 'IMG_MIR_TEMP',
                'count_TIR1': 'IMG_TIR1',
                'temp_TIR1': 'IMG_TIR1_TEMP',
                'count_TIR2': 'IMG_TIR2',
                'temp_TIR2': 'IMG_TIR2_TEMP',
                'count_WV': 'IMG_WV',
                'temp_WV': 'IMG_WV_TEMP',
                'St_E': 'Sat_Elevation',
                'Sn_E': 'Sun_Elevation'
            }
            
            for key, path in datasets.items():
                try:
                    if path in f:
                        arr = np.array(f[path], dtype=np.float64)
                        
                        # Handle 3D arrays by using the first layer
                        if arr.ndim == 3:
                            arr = np.squeeze(arr, axis=0)
                        
                        # Handle special elevation values
                        if path in ['Sat_Elevation', 'Sun_Elevation']:
                            arr[arr == 32767] = np.nan
                            arr = arr * 0.01
                            
                        data[key] = arr
                    else:
                        print(f"Warning: Dataset {path} not found")
                        data[key] = None
                except Exception as e:
                    print(f"Error reading dataset {path}: {str(e)}")
                    data[key] = None
                    
            return data
    except Exception as e:
        print(f"Error reading HDF5 file: {str(e)}")
        return None


def BCT_method(BTD_TM_bct, CNDI, CPDI, ms, ms_geoloc):
    """Find clouds using BCT method with proper dimension handling and boundary checks"""    
    if BTD_TM_bct is None or np.all(np.isnan(BTD_TM_bct)):
        print("Warning: All BTD_TM_bct values are NaN")
        return np.zeros_like(BTD_TM_bct) if BTD_TM_bct is not None else np.array([])
    
    m, n = BTD_TM_bct.shape
    print(f"Array dimensions: m={m}, n={n}")
    # Mask snow covered areas
    BTD_TM_bct = np.where(ms_geoloc == 5, np.nan, BTD_TM_bct)
    
    BCT_mask1 = np.zeros((m, n))
    
    for p in range(2):  # ocean/land mask condition (0, 1)
        # Create mask for current land/ocean type
        # Fix: The original MATLAB code sets BTD_TM_n(ms==p)=nan, meaning keep values where ms != p
        mask = (ms != p)  # This was the issue - should be != not ==
        BTD_TM_n = np.where(mask, BTD_TM_bct.copy(), np.nan)
        
        for i in range(m-1):  # i goes from 0 to m-2, so i+1 goes from 1 to m-1
            for j in range(n):  # j goes from 0 to n-1, which is correct
                # Check bounds explicitly to be safe
                if i+1 >= m or j >= n:
                    print(f"Skipping out of bounds: i+1={i+1}, j={j}, m={m}, n={n}")
                    continue
                    
                if np.isnan(BTD_TM_n[i,j]) or np.isnan(BTD_TM_n[i+1,j]):
                    continue
                    
                # Calculate variance
                try:
                    var = np.nanvar([BTD_TM_n[i,j], BTD_TM_n[i+1,j]])
                    
                    if var > 7.25:
                        BCT_mask1[i+1,j] += 1
                    else:
                        if BCT_mask1[i,j] == 1:
                            diff = abs(BTD_TM_n[i+1,j]) - abs(BTD_TM_n[i,j])
                            if diff >= 0:
                                BCT_mask1[i+1,j] += 1
                        elif BCT_mask1[i,j] == 0:
                            diff = BTD_TM_n[i+1,j] - BTD_TM_n[i,j]
                            # Fix: The original MATLAB has (2/3)*3 which equals 2
                            if diff < -3 or diff > 2:
                                BCT_mask1[i+1,j] += 1
                    
                    # Additional threshold checks when not already classified
                    if BCT_mask1[i+1,j] != 1:
                        if BTD_TM_n[i+1,j] < 0:
                            if BTD_TM_n[i+1,j] - CNDI[i+1,j] < -4:
                                BCT_mask1[i+1,j] += 1
                        else:
                            if BTD_TM_n[i+1,j] - CPDI[i+1,j] > 2.5:
                                BCT_mask1[i+1,j] += 1
                                
                except IndexError as e:
                    print(f"Index error at i={i}, j={j}, i+1={i+1}: {e}")
                    continue
                except Exception as e:
                    print(f"Error in BCT computation at [{i},{j}]: {e}")
                    continue
    
    return BCT_mask1

def pre_day_composite(pth, fnm, dd):
    """Previous 30 days composite with robust error handling and date processing"""
    m, n = 1616, 1618  # Standard array dimensions
    nd = 30  # Number of days to look back
    
    # Initialize arrays
    t11 = np.full((m, n, nd), np.nan)
    diff_neg = np.full((m, n, nd), np.nan)
    diff_pos = np.full((m, n, nd), np.nan)
    valid_days = 0
    
    # Extract original date pattern from filename
    try:
        # Parse original date pattern
        date_format = "%d%b%Y_%H%M"
        original_date_part = fnm[6:20]
        dd = datetime.datetime.strptime(original_date_part, date_format)
    except Exception as e:
        print(f"Error parsing date from filename: {e}")
        # Return empty arrays if date parsing fails
        return np.zeros((m,n)), np.zeros((m,n)), np.zeros((m,n))
    
    for k in range(1, nd+1):
        # Calculate previous date
        try:
            dm = dd - datetime.timedelta(days=(31-k))
            yn = dm.strftime('%Y')
            mn = dm.strftime('%b').upper()
            dn = dm.strftime('%d')
        except Exception as e:
            print(f"Error calculating date: {e}")
            continue
        
        # Try multiple time offsets for more robust file finding
        time_offsets = [0, -1, -2, 1, 2]  # Minutes offset to try
        
        found = False
        for offset in time_offsets:
            try:
                curr_time = dm + datetime.timedelta(minutes=offset)
                time_str = curr_time.strftime('%H%M')
                
                # Construct filename with properly formatted components
                base_fnm = fnm[:6] + dn + mn + yn + '_' + time_str
                
                # Ensure the extension part from the original filename is preserved
                if len(fnm) > 20:
                    extension_part = fnm[20:]
                else:
                    extension_part = '.h5'  # Default extension if original is too short
                
                fnn = os.path.join(pth, base_fnm + extension_part)
                
                if os.path.exists(fnn):
                    try:
                        with h5py.File(fnn, 'r') as f:
                            if 'IMG_TIR1' in f and 'IMG_TIR1_TEMP' in f and 'IMG_MIR' in f and 'IMG_MIR_TEMP' in f:
                                # Read data
                                count_TIR1 = np.array(f['IMG_TIR1'], dtype=float)
                                temp_TIR1 = np.array(f['IMG_TIR1_TEMP'], dtype=float)
                                count_MIR = np.array(f['IMG_MIR'], dtype=float)
                                temp_MIR = np.array(f['IMG_MIR_TEMP'], dtype=float)
                                
                                # Handle 3D arrays
                                if count_TIR1.ndim == 3:
                                    count_TIR1 = np.squeeze(count_TIR1, axis=0)
                                if count_MIR.ndim == 3:
                                    count_MIR = np.squeeze(count_MIR, axis=0)
                                
                                # Process data safely
                                t00 = np.full((m, n), np.nan)
                                t_mir = np.full((m, n), np.nan)
                                
                                # Ensure dimensions match
                                count_TIR1 = count_TIR1[:m, :n] if count_TIR1.shape[0] >= m and count_TIR1.shape[1] >= n else np.pad(
                                    count_TIR1, 
                                    ((0, max(0, m - count_TIR1.shape[0])), (0, max(0, n - count_TIR1.shape[1]))),
                                    'constant', 
                                    constant_values=np.nan
                                )
                                
                                count_MIR = count_MIR[:m, :n] if count_MIR.shape[0] >= m and count_MIR.shape[1] >= n else np.pad(
                                    count_MIR, 
                                    ((0, max(0, m - count_MIR.shape[0])), (0, max(0, n - count_MIR.shape[1]))),
                                    'constant', 
                                    constant_values=np.nan
                                )
                                
                                # Vectorized operations for performance
                                valid_TIR1 = (count_TIR1 != 0) & (count_TIR1 != 1023) & (~np.isnan(count_TIR1))
                                valid_MIR = (count_MIR != 0) & (count_MIR != 1023) & (~np.isnan(count_MIR))
                                
                                # Safely convert to integer indices
                                tir1_indices = count_TIR1[valid_TIR1].astype(int)
                                mir_indices = count_MIR[valid_MIR].astype(int)
                                
                                # Protect against out-of-bounds indexing
                                tir1_indices = np.clip(tir1_indices, 0, temp_TIR1.shape[0]-1)
                                mir_indices = np.clip(mir_indices, 0, temp_MIR.shape[0]-1)
                                
                                # Get temperature values
                                t00_values = temp_TIR1[tir1_indices, 0] if len(tir1_indices) > 0 else []
                                t_mir_values = temp_MIR[mir_indices, 0] if len(mir_indices) > 0 else []
                                
                                # Set values in result arrays
                                valid_coords_TIR1 = np.where(valid_TIR1)
                                valid_coords_MIR = np.where(valid_MIR)
                                
                                # Set values only where valid
                                if len(valid_coords_TIR1[0]) == len(t00_values):
                                    t00[valid_coords_TIR1] = t00_values
                                    
                                if len(valid_coords_MIR[0]) == len(t_mir_values):
                                    t_mir[valid_coords_MIR] = t_mir_values
                                
                                # Calculate difference
                                diff = t00 - t_mir
                                
                                # Store results
                                t11[:, :, valid_days] = t00
                                diff_pos[:, :, valid_days] = np.where(diff > 0, diff, np.nan)
                                diff_neg[:, :, valid_days] = np.where(diff < 0, diff, np.nan)
                                
                                valid_days += 1
                                found = True
                                print(f"Found valid file for {dm.strftime('%d-%b-%Y')}")
                                break
                            else:
                                print(f"Missing required datasets in {fnn}")
                    except Exception as e:
                        print(f"Error processing {fnn}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error constructing filename: {e}")
                continue
        
        if not found:
            print(f"No valid file found for {dm.strftime('%d-%b-%Y')}")
    
    # Calculate composites only if we have valid data
    if valid_days ==-0:
        print("Warning: No valid days found for composite")
        return np.zeros((m,n)), np.zeros((m,n)), np.zeros((m,n))
    
    # Use only valid days for calculations
    BTmx = np.nanmax(t11[:,:,:valid_days], axis=2)
    
    # Handle potential all-NaN slices
    CPDI = np.zeros((m, n))
    CNDI = np.zeros((m, n))
    
    # Calculate min/max safely
    for i in range(m):
        for j in range(n):
            pos_vals = diff_pos[i, j, :valid_days]
            neg_vals = diff_neg[i, j, :valid_days]
            
            if np.any(~np.isnan(pos_vals)):
                CPDI[i, j] = np.nanmin(pos_vals)
            
            if np.any(~np.isnan(neg_vals)):
                CNDI[i, j] = np.nanmax(neg_vals)
    
    return BTmx, CPDI, CNDI

def Spatial_coherence_sd_fast(t11, BTD_TM, ms, ms_geoloc):
    """
    Fast version using scipy.ndimage for sliding window operations
    """
    import numpy as np
    from scipy import ndimage
    
    print("Performing Fast Spatial Coherence")
    
    # Handle dimensionality
    is_t11_3d = len(t11.shape) == 3
    is_BTD_3d = len(BTD_TM.shape) == 3
    
    def fast_std_3x3(arr, mask):
        """Calculate 3x3 std using uniform filter"""
        # Mask invalid areas
        arr_masked = np.where(mask, arr, np.nan)
        
        # Use uniform filter for mean calculation
        # Handle NaN values by replacing with 0 and tracking valid counts
        arr_zero = np.nan_to_num(arr_masked, 0)
        valid = ~np.isnan(arr_masked)
        
        # Calculate sums and counts in 3x3 windows
        sum_vals = ndimage.uniform_filter(arr_zero.astype(float), size=3, mode='constant')
        count_vals = ndimage.uniform_filter(valid.astype(float), size=3, mode='constant')
        
        # Calculate means
        mean_vals = np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals>0)
        
        # Calculate variance using E[X²] - E[X]²
        arr_sq = arr_zero ** 2
        sum_sq = ndimage.uniform_filter(arr_sq, size=3, mode='constant')
        mean_sq = np.divide(sum_sq, count_vals, out=np.zeros_like(sum_sq), where=count_vals>0)
        
        variance = mean_sq - mean_vals**2
        std_vals = np.sqrt(np.maximum(variance, 0))
        
        # Only return std where we have at least 2 valid pixels
        return np.where(count_vals >= 2, std_vals, np.nan)
    
    if is_t11_3d:
        SD_t11 = np.full_like(t11, np.nan)
        if is_BTD_3d:
            SD_BTD_TM = np.full_like(BTD_TM, np.nan)
        else:
            SD_BTD_TM = np.full_like(BTD_TM, np.nan)
        
        for band in range(t11.shape[2]):
            print(f"Processing band {band + 1}/{t11.shape[2]}")
            
            t11_2d = t11[:, :, band]
            BTD_TM_2d = BTD_TM if not is_BTD_3d else BTD_TM[:, :, band]
            
            # Process each condition
            sd_t11_band = np.full_like(t11_2d, np.nan)
            sd_btd_band = np.full_like(BTD_TM_2d, np.nan)
            
            for k in range(1, 5):
                for p in [0, 1]:
                    mask = (ms_geoloc == k) & (ms == p) & (ms_geoloc != 5)
                    if np.any(mask):
                        std_t11 = fast_std_3x3(t11_2d, mask)
                        std_btd = fast_std_3x3(BTD_TM_2d, mask)
                        
                        sd_t11_band = np.where(mask, std_t11, sd_t11_band)
                        sd_btd_band = np.where(mask, std_btd, sd_btd_band)
            
            SD_t11[:, :, band] = sd_t11_band
            if is_BTD_3d:
                SD_BTD_TM[:, :, band] = sd_btd_band
            elif band == 0:
                SD_BTD_TM = sd_btd_band
    else:
        # 2D processing
        SD_t11 = np.full_like(t11, np.nan)
        SD_BTD_TM = np.full_like(BTD_TM, np.nan)
        
        for k in range(1, 5):
            for p in [0, 1]:
                mask = (ms_geoloc == k) & (ms == p) & (ms_geoloc != 5)
                if np.any(mask):
                    std_t11 = fast_std_3x3(t11, mask)
                    std_btd = fast_std_3x3(BTD_TM, mask)
                    
                    SD_t11 = np.where(mask, std_t11, SD_t11)
                    SD_BTD_TM = np.where(mask, std_btd, SD_BTD_TM)
    
    return SD_t11, SD_BTD_TM


def snow_test(pth, fnm, topo_N, ms):
    """Snow test function with improved array handling and error detection"""
    print("Running snow detection test")
    m = 1616
    n = 1618
    # Initialize arrays with proper dimensions
    CM_s = np.zeros((m, n))
    t_mir = np.full((m, n), np.nan)
    t11 = np.full((m, n), np.nan)
    t12 = np.full((m, n), np.nan)
    
    # Adjust filename to 0730 time (reference time for snow detection)
    fnm1 = f"{fnm[:16]}0730{fnm[20:]}"
    
    try:
        # Parse date from filename
        date_format = "%d%b%Y_%H%M"
        dd = datetime.datetime.strptime(fnm[6:20], date_format)
        
        # Generate time range with ±2 minutes to handle time offsets
        base_time = dd
        time_range = [base_time + datetime.timedelta(minutes=i-2) for i in range(5)]
        
        found_file = False
        for curr_time in time_range:
            time_str = curr_time.strftime('%H%M')
            fnn = os.path.join(pth, f"{fnm[:6]}{dd.strftime('%d%b%Y')}_{time_str}{fnm[20:]}")
            
            if os.path.exists(fnn):
                found_file = True
                with h5py.File(fnn, 'r') as f:
                    # Read and properly handle arrays
                    try:
                        # Get count arrays and handle possible dimensions
                        count_MIR = np.array(f['/IMG_MIR'], dtype=float)
                        if count_MIR.ndim == 3:
                            count_MIR = np.squeeze(count_MIR, axis=0)
                            
                        count_TIR1 = np.array(f['/IMG_TIR1'], dtype=float)
                        if count_TIR1.ndim == 3:
                            count_TIR1 = np.squeeze(count_TIR1, axis=0)
                            
                        count_TIR2 = np.array(f['/IMG_TIR2'], dtype=float)
                        if count_TIR2.ndim == 3:
                            count_TIR2 = np.squeeze(count_TIR2, axis=0)
                        
                        # Get temperature LUTs
                        temp_MIR = np.array(f['/IMG_MIR_TEMP'], dtype=float)
                        temp_TIR1 = np.array(f['/IMG_TIR1_TEMP'], dtype=float)
                        temp_TIR2 = np.array(f['/IMG_TIR2_TEMP'], dtype=float)
                        
                        # Get sun elevation and ensure proper dimensions
                        Sn_E = np.array(f['/Sun_Elevation'], dtype=float)
                        if Sn_E.ndim == 3:
                            Sn_E = np.squeeze(Sn_E, axis=0)
                        
                        # Process Sun Elevation
                        Sn_E[Sn_E == 32767] = np.nan
                        Sn_E = Sn_E * 0.01
                        Sn_z = 90 - Sn_E
                        # Initialize snow mask where sun elevation > 10 degrees
                        CM_s = np.zeros((m, n))
                        valid_sun = Sn_E > 10
                        CM_s[valid_sun] = 0
                        CM_s[ms == 0] = np.nan  # No snow detection on ocean
                        
                        # Process MIR channel
                        valid_MIR_indices = np.where((count_MIR > 0) & (count_MIR < 1023))
                        for i, j in zip(*valid_MIR_indices):
                            if i < m and j < n:  # Boundary check
                                idx = int(count_MIR[i, j])
                                if idx < len(temp_MIR):  # Ensure index is valid
                                    t_mir[i, j] = temp_MIR[idx]
                        
                        # Process TIR1 channel
                        valid_TIR1_indices = np.where((count_TIR1 > 0) & (count_TIR1 < 1023))
                        for i, j in zip(*valid_TIR1_indices):
                            if i < m and j < n:  # Boundary check
                                idx = int(count_TIR1[i, j])
                                if idx < len(temp_TIR1):  # Ensure index is valid
                                    t11[i, j] = temp_TIR1[idx]
                        
                        # Process TIR2 channel
                        valid_TIR2_indices = np.where((count_TIR2 > 0) & (count_TIR2 < 1023))
                        for i, j in zip(*valid_TIR2_indices):
                            if i < m and j < n:  # Boundary check
                                idx = int(count_TIR2[i, j])
                                if idx < len(temp_TIR2):  # Ensure index is valid
                                    t12[i, j] = temp_TIR2[idx]
                        
                        # Apply land mask
                        t11[ms == 0] = np.nan
                        t12[ms == 0] = np.nan
                        t_mir[ms == 0] = np.nan
                        
                        # Calculate brightness temperature differences
                        BTD_TT = t11 - t12
                        BTD_TM = t11 - t_mir
                        
                        # Handle division by zero in normalization
                        Sn_z_rad = Sn_z * (np.pi/180)
                        cos_Sn_z = np.cos(Sn_z_rad)
                        cos_Sn_z[cos_Sn_z == 0] = np.nan  # Avoid division by zero
                        BTD_TM_n = BTD_TM / cos_Sn_z
                        
                        topo_thr = 300 - ((topo_N * 10) / 1000)
                        
                        topo_thr[ms == 0] = np.nan
                        
                        # Apply snow test conditions with vectorized operations
                        snow_cond1 = (BTD_TM_n < 10) & ~np.isnan(BTD_TM_n)
                        snow_cond2 = (t11 < 286.15) & (t11 > topo_thr - 5) & ~np.isnan(t11) & ~np.isnan(topo_thr)
                        snow_cond3 = (BTD_TT < 2) & ~np.isnan(BTD_TT)
                        
                        # Update snow mask
                        CM_s[snow_cond1] += 1
                        CM_s[snow_cond2] += 1
                        CM_s[snow_cond3] += 1
                    
                    except Exception as e:
                        print(f"Error processing data in snow_test: {str(e)}")
                break  # Found and processed a file, exit the loop
        
        if not found_file:
            print(f"No valid file found for snow detection at time range around {dd}")
    
    except Exception as e:
        print(f"Error in snow_test: {str(e)}")
        return np.zeros((m, n))
    
    # Create final snow mask - consider snow where at least 2 conditions were met
    SM = np.zeros((m, n))
    SM[CM_s > 2] = 1
    
    return SM

def CM_4km_generation_n(path, Fname, ms, sst_D, sst_N, topo_N):
    """Cloud mask generation with corrected dimension handling"""
    print(f"Generating cloud mask for {Fname}")
    
    # Ensure ms has appropriate orientation
    if ms.shape[0] != topo_N.shape[0] or ms.shape[1] != topo_N.shape[1]:
        print(f"Adjusting mask dimensions from {ms.shape} to {topo_N.shape}")
        if ms.shape[0] == topo_N.shape[1] and ms.shape[1] == topo_N.shape[0]:
            ms = ms.T
        else:
            # If dimensions don't match even after transpose, try to resize
            temp_ms = np.zeros(topo_N.shape)
            min_rows = min(ms.shape[0], topo_N.shape[0])
            min_cols = min(ms.shape[1], topo_N.shape[1])
            temp_ms[:min_rows, :min_cols] = ms[:min_rows, :min_cols]
            ms = temp_ms
    
    # Validate input file
    if not validate_h5_file(os.path.join(path, Fname)):
        print(f"File validation failed for {Fname}")
        return np.zeros_like(ms), np.zeros_like(ms)
    
    # Read data with robust reader
    data = read_h5_data(os.path.join(path, Fname))
    if data is None or 'count_TIR1' not in data or data['count_TIR1'] is None:
        print(f"Failed to read required data from {Fname}")
        return np.zeros_like(ms), np.zeros_like(ms)
    
    # Get dimensions from count arrays (these should be 2D)
    m, n = data['count_TIR1'].shape
    print(f"Data dimensions: {m} x {n}")
    
    # Parse date from filename
    try:
        dd = datetime.datetime.strptime(Fname[6:20], "%d%b%Y_%H%M")
        mon = dd.month - 1  # Convert to 0-based indexing
    except ValueError as e:
        print(f"Error parsing date from filename {Fname}: {str(e)}")
        return np.zeros_like(ms), np.zeros_like(ms)
    
    # Initialize 2D output arrays (NOT 3D)
    R_vis = np.full((m, n), np.nan)
    t_mir = np.full((m, n), np.nan)
    t_wv = np.full((m, n), np.nan)
    t11 = np.full((m, n), np.nan)  # This should be 2D
    t12 = np.full((m, n), np.nan)
    
    # Process each channel data - CORRECTED VERSION
    try:
        # Process VIS channel
        if data['count_VIS'] is not None and data['rad_VIS'] is not None:
            count_vis = data['count_VIS']
            rad_vis = data['rad_VIS']
            
            # Loop through each pixel
            for i in range(m):
                for j in range(n):
                    count_val = count_vis[i, j]
                    if not np.isnan(count_val) and count_val > 0 and count_val < len(rad_vis):
                        idx = int(count_val) - 1  # Convert to 0-based indexing
                        if idx >= 0 and idx < len(rad_vis):
                            R_vis[i, j] = rad_vis[idx]
        
        # Process MIR channel
        if data['count_MIR'] is not None and data['temp_MIR'] is not None:
            count_mir = data['count_MIR']
            temp_mir = data['temp_MIR']
            
            for i in range(m):
                for j in range(n):
                    count_val = count_mir[i, j]
                    if not np.isnan(count_val) and count_val > 0 and count_val != 1023:
                        idx = int(count_val) - 1  # Convert to 0-based indexing
                        if idx >= 0 and idx < len(temp_mir):
                            t_mir[i, j] = temp_mir[idx]
        
        # Process WV channel
        if data['count_WV'] is not None and data['temp_WV'] is not None:
            count_wv = data['count_WV']
            temp_wv = data['temp_WV']
            
            for i in range(m):
                for j in range(n):
                    count_val = count_wv[i, j]
                    if not np.isnan(count_val) and count_val > 0 and count_val != 1023:
                        idx = int(count_val) - 1  # Convert to 0-based indexing
                        if idx >= 0 and idx < len(temp_wv):
                            t_wv[i, j] = temp_wv[idx]
        
        # Process TIR1 channel - THIS IS THE KEY FIX
        if data['count_TIR1'] is not None and data['temp_TIR1'] is not None:
            count_tir1 = data['count_TIR1']
            temp_tir1 = data['temp_TIR1']
            
            for i in range(m):
                for j in range(n):
                    count_val = count_tir1[i, j]
                    if not np.isnan(count_val) and count_val > 0 and count_val != 1023:
                        idx = int(count_val) - 1  # Convert to 0-based indexing
                        if idx >= 0 and idx < len(temp_tir1):
                            t11[i, j] = temp_tir1[idx]  # This creates a 2D array
        
        # Process TIR2 channel
        if data['count_TIR2'] is not None and data['temp_TIR2'] is not None:
            count_tir2 = data['count_TIR2']
            temp_tir2 = data['temp_TIR2']
            
            for i in range(m):
                for j in range(n):
                    count_val = count_tir2[i, j]
                    if not np.isnan(count_val) and count_val > 0 and count_val != 1023:
                        idx = int(count_val) - 1  # Convert to 0-based indexing
                        if idx >= 0 and idx < len(temp_tir2):
                            t12[i, j] = temp_tir2[idx]
    
    except Exception as e:
        print(f"Error processing channel data: {str(e)}")
        return np.zeros_like(ms), np.zeros_like(ms)
    
    # Verify dimensions before continuing
    print(f"t11 shape: {t11.shape}")
    print(f"t_mir shape: {t_mir.shape}")
    
    # Calculate brightness temperature differences
    BTD_TT = t11 - t12
    BTD_TM = t11 - t_mir
    BTD_TW = t11 - t_wv
    
    # Generate geolocation mask
    ms_geoloc = np.full((m, n), np.nan)
    
    # Process elevation data
    Sn_E = data['Sn_E'] if data['Sn_E'] is not None else np.zeros((m, n))
    St_E = data['St_E'] if data['St_E'] is not None else np.zeros((m, n))
    
    # Ensure elevation arrays have correct dimensions
    if Sn_E.shape != (m, n):
        print(f"Resizing Sn_E from {Sn_E.shape} to {(m, n)}")
        temp_sn_e = np.zeros((m, n))
        min_rows = min(Sn_E.shape[0], m)
        min_cols = min(Sn_E.shape[1], n)
        temp_sn_e[:min_rows, :min_cols] = Sn_E[:min_rows, :min_cols]
        Sn_E = temp_sn_e
    
    if St_E.shape != (m, n):
        print(f"Resizing St_E from {St_E.shape} to {(m, n)}")
        temp_st_e = np.zeros((m, n))
        min_rows = min(St_E.shape[0], m)
        min_cols = min(St_E.shape[1], n)
        temp_st_e[:min_rows, :min_cols] = St_E[:min_rows, :min_cols]
        St_E = temp_st_e
    
    # Use vectorized operations for geolocation mask
    ms_geoloc[Sn_E > 10] = 1  # Daytime
    ms_geoloc[(Sn_E > 0) & (Sn_E <= 10)] = 2  # Twilight
    ms_geoloc[Sn_E == 0] = 3  # Nighttime
    
    # Sunglint identification
    Sn_z = 90 - Sn_E
    St_z = 90 - St_E
    
    Zn = Sn_z + St_z
    Zn_ocean = Zn.copy()
    Zn_ocean[ms != 0] = np.nan  # Only ocean pixels
    
    # Calculate sunglint probability
    with np.errstate(invalid='ignore'):
        Pb_SG = 100 * np.exp(-((Zn_ocean / 8.5) ** 2) / 2)
    ms_geoloc[Pb_SG > 0.1] = 4  # Sunglint
    
    # Snow detection
    try:
        ms_sn = snow_test(path, Fname, topo_N, ms)
        ms_geoloc[ms_sn == 1] = 5  # Snow
    except Exception as e:
        print(f"Warning: Snow test failed: {str(e)}")
    
    # Get composites from previous 30 days
    try:
        BTmx, CPDI, CNDI = pre_day_composite(path, Fname, dd)
        print(f"BTmx shape: {BTmx.shape}")
    except Exception as e:
        print(f"Warning: Composite generation failed: {str(e)}")
        BTmx = np.full((m, n), 280.0)  # Default temperature
        CPDI = np.zeros((m, n))
        CNDI = np.zeros((m, n))
    
    # Spatial coherence
    try:
        SD_t11, SD_BTD_TM = Spatial_coherence_sd_fast(t11, BTD_TM, ms, ms_geoloc)
    except Exception as e:
        print(f"Warning: Spatial coherence failed: {str(e)}")
        SD_t11 = np.zeros((m, n))
        SD_BTD_TM = np.zeros((m, n))
    
    # Ensure CNDI and CPDI have proper dimensions
    if CNDI.shape != (m, n):
        print(f"Resizing CNDI from {CNDI.shape} to {(m, n)}")
        temp_CNDI = np.full((m, n), np.nan)
        min_rows = min(CNDI.shape[0], m)
        min_cols = min(CNDI.shape[1], n)
        temp_CNDI[:min_rows, :min_cols] = CNDI[:min_rows, :min_cols]
        CNDI = temp_CNDI
    
    if CPDI.shape != (m, n):
        print(f"Resizing CPDI from {CPDI.shape} to {(m, n)}")
        temp_CPDI = np.full((m, n), np.nan)
        min_rows = min(CPDI.shape[0], m)
        min_cols = min(CPDI.shape[1], n)
        temp_CPDI[:min_rows, :min_cols] = CPDI[:min_rows, :min_cols]
        CPDI = temp_CPDI
    
    # BCT method
    try:
        BCT_mask = BCT_method(BTD_TM, CNDI, CPDI, ms, ms_geoloc)
    except Exception as e:
        print(f"Warning: BCT method failed: {str(e)}")
        BCT_mask = np.zeros((m, n))
    
    # Initialize cloud mask - NOW t11 and BTmx should have compatible shapes
    CLD_mask_n = np.zeros((m, n))
    
    # Apply detection tests
    print(f"Applying cloud detection tests...")
    print(f"t11 shape: {t11.shape}, BTmx shape: {BTmx.shape}")
    
    # 1. Primary test using BTmx threshold
    if BTmx.shape == (m, n):
        mask_condition = (t11 < BTmx - 5) & ~np.isnan(t11) & ~np.isnan(BTmx)
        CLD_mask_n[mask_condition] = 1
        print(f"Primary test: {np.sum(mask_condition)} pixels detected")
    else:
        print(f"BTmx shape mismatch: {BTmx.shape} vs expected {(m, n)}")
    
    # 2. BCT test results
    if BCT_mask.shape == (m, n):
        mask_condition = BCT_mask > 0
        CLD_mask_n[mask_condition] = 1
        print(f"BCT test: {np.sum(mask_condition)} pixels detected")
    else:
        print(f"BCT_mask shape mismatch: {BCT_mask.shape} vs expected {(m, n)}")
    
    # 3. Spatial coherence test
    if SD_t11.shape == (m, n):
        mask_condition = SD_t11 > 1.5
        CLD_mask_n[mask_condition] = 1
        print(f"Spatial coherence t11: {np.sum(mask_condition)} pixels detected")
    else:
        print(f"SD_t11 shape mismatch: {SD_t11.shape} vs expected {(m, n)}")
    
    if SD_BTD_TM.shape == (m, n):
        mask_condition = SD_BTD_TM > 1.0
        CLD_mask_n[mask_condition] = 1
        print(f"Spatial coherence BTD_TM: {np.sum(mask_condition)} pixels detected")
    else:
        print(f"SD_BTD_TM shape mismatch: {SD_BTD_TM.shape} vs expected {(m, n)}")
    
    # Calculate cloud fraction
    CF_mask_n = np.zeros((m, n))
    CF_mask_n[CLD_mask_n == 1] = 1
    
    print(f"Cloud detection complete. Total cloud pixels: {np.sum(CLD_mask_n == 1)}")
    
    return CLD_mask_n, CF_mask_n

def main():
    """
    Main function to process INSAT-3D data and generate cloud mask products.
    Python equivalent of CM_4km_call_n.m with improved error handling and dimension consistency.
    """
    import os
    import glob
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import netCDF4 as netcdf
    from scipy.io import loadmat
    import imageio
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='cloud_detection.log'
    )
    logger = logging.getLogger()
    
    # Set up working directory
    curdir = os.getcwd()
    os.chdir(curdir)
    txtfile = '/Users/harshrajsahu/Desktop/cloud/Py_Cloud/CM_locs4km.txt'

    with open(txtfile, 'r') as f:
        cc = f.readlines()

    
    # Read configuration file with error handling
    try:
        with open(txtfile, 'r') as f:
            cc = f.readlines()
        
        if len(cc) < 3:
            print('\n\t Error: Configuration file does not have enough lines')
            return
            
        # Parse configuration
        cttloc = cc[0].strip()  # Input path for HDF5 files
        outloc = cc[1].strip()  # Output path for results
        check = int(cc[2].strip())  # Date selection option
        
        # Create output directory if it doesn't exist
        os.makedirs(outloc, exist_ok=True)
        
    except FileNotFoundError:
        print(f'\n\t Error: Configuration file {txtfile} not found')
        return
    except ValueError:
        print('\n\t Error: Invalid format in configuration file')
        return
    
    # Determine processing date
    if check == 1:
        # Use yesterday's date
        req = datetime.datetime.now() - datetime.timedelta(days=1)
        today_date = req.strftime('%d%b%Y').upper()
    elif check == 2:
        # Use specified date
        if len(cc) < 4:
            print('\n\t Error: No date specified in configuration file')
            return
        today_date = cc[3].strip()
    else:
        print('\n\t Wrong Input - check must be 1 or 2')
        return
    
    print(f'\n\t Processing date: {today_date}')
    
    # Verify paths
    if not os.path.exists(cttloc):
        print(f'\n\t Error: Input path {cttloc} does not exist')
        return
    
    print('\n\t Reading Files...')
    
    # Find all HDF5 files for the specified date
    fnD = glob.glob(os.path.join(cttloc, f'*{today_date}*.h5'))
    
    if not fnD:
        print(f'\n\t No HDF5 files found for date {today_date}')
        return
    
    fnm = [os.path.basename(f) for f in fnD]
    print(f'\n\t Found {len(fnm)} files to process')
    
    # Load ancillary data with error handling
    try:
        # Load geographic data
        lnlt_data = loadmat('/Users/harshrajsahu/Desktop/cloud/Py_Cloud/INSAT3D_MER_IND_lnlt.mat')
        ln = lnlt_data.get('ln')
        ln=np.transpose(ln)# Longitude
        lt = lnlt_data.get('lt')
        lt=np.transpose(lt)
        
        if ln is None or lt is None:
            print('\n\t Error: Missing longitude/latitude data')
            return
        
        # Load mask data
        mask_data = loadmat('/Users/harshrajsahu/Desktop/cloud/Py_Cloud/mask.mat')
        ms = mask_data.get('ms')
        ms=np.transpose(ms)# Land/sea mask
        
        if ms is None:
            print('\n\t Error: Missing land/sea mask data')
            return
        
        # Load topography data
        topo_data = loadmat('/Users/harshrajsahu/Desktop/cloud/Py_Cloud/topo_IND_4km.mat')
        topo_N = topo_data.get('topo_N')  # Topography
        topo_N = topo_N.T
        if topo_N is None:
            print('\n\t Error: Missing topography data')
            return
        
        # Load SST data
        sst_day_data = loadmat('/Users/harshrajsahu/Desktop/cloud/Py_Cloud/sst_mclim_terra_day_IND.mat')
        sst_D = sst_day_data.get('sst_D')  # Day SST
        sst_D = sst_D.swapaxes(0,1)
        
        sst_night_data = loadmat('/Users/harshrajsahu/Desktop/cloud/Py_Cloud/sst_mclim_terra_night_IND.mat')
        sst_N = sst_night_data.get('sst_N')  # Night SST
        sst_N = sst_N.swapaxes(0,1)
        
        if sst_D is None or sst_N is None:
            print('\n\t Error: Missing SST data')
            return
            
    except Exception as e:
        print(f'\n\t Error loading ancillary data: {str(e)}')
        return
    
    # Verify and standardize data dimensions
    m, n = ms.shape
    # Initialize arrays for cloud mask and flag data
    CM_4kmG = np.zeros((m, n, len(fnm)))  # Cloud mask 3D array
    CFlag = np.zeros((m, n, len(fnm)))    # Cloud flag 3D array
    t = np.zeros((len(fnm),))             # Time values (1D array)
    
    # Process each file
    for i, Fname in enumerate(fnm):
        print(f'\n\t Processing file {i+1}/{len(fnm)}: {Fname}')
        oname_n = f'CTT_I3D_{Fname[6:20]}.mat'
        matfile = os.path.join(outloc, oname_n)
        
        # Check if file already processed
        if os.path.exists(matfile):
            print(f'\t File already processed. Loading from {matfile}')
            try:
                mat_data = loadmat(matfile)
                CM_4kmG[:, :, i] = mat_data.get('CLD_mask_4km', np.zeros((m, n))).T
                CFlag[:, :, i] = mat_data.get('CF_4km', np.zeros((m, n))).T
                
                # Extract time from filename (in format HHMM)
                time_str = Fname[16:20]
                t[i] = int(time_str)
                continue
            except Exception as e:
                print(f'\t Error loading existing file: {str(e)}')
                print('\t Processing file from scratch')
        
        try:
            # Process the file to generate cloud mask and flag
            CLD_mask_4km, CF_4km = CM_4km_generation_n(cttloc, Fname, ms, sst_D, sst_N, topo_N)
            
            
            # Store results
            CM_4kmG[:, :, i] = CLD_mask_4km
            CFlag[:, :, i] = CF_4km
            
            # Extract time from filename (in format HHMM)
            time_str = Fname[16:20]
            t[i] = int(time_str)
            
            # Save processed file
            try:
                import scipy.io as sio
                sio.savemat(matfile, {'CLD_mask_4km': CLD_mask_4km, 'CF_4km': CF_4km})
                print(f'\t Saved processed data to {matfile}')
            except Exception as e:
                print(f'\t Warning: Could not save .mat file: {str(e)}')
                
        except Exception as e:
            print(f'\t Error processing file: {str(e)}')
            # Fill with zeros if processing fails
            CM_4kmG[:, :, i] = np.zeros((m, n))
            CFlag[:, :, i] = np.zeros((m, n))
    
# Fixed NetCDF writing section - replace the NetCDF part in your main() function

    # Create NetCDF output file
    print('\n\t Writing output in NetCDF format')
    
    # Get the actual data dimensions
    m, n = ms.shape  # This should match your cloud mask data dimensions
    
    # Create proper coordinate arrays based on your geographic data
    # Since ln and lt are 2D arrays representing the grid coordinates
    # Extract 1D coordinate arrays from them
    
    # For latitude: take the first column (constant longitude, varying latitude)
    latitude = lt[:, 0]  # First column gives latitude variation
    
    # For longitude: take the first row (constant latitude, varying longitude)  
    longitude = ln[0, :]  # First row gives longitude variation
    
    # Remove any duplicates and sort (though they should already be sorted)
    latitude = np.unique(latitude)
    longitude = np.unique(longitude)
    latitude.sort()
    longitude.sort()
    
    print(f'\t Grid dimensions: {m} x {n}')
    print(f'\t Latitude range: {latitude.min():.2f} to {latitude.max():.2f}')
    print(f'\t Longitude range: {longitude.min():.2f} to {longitude.max():.2f}')
    
    # Create NetCDF file
    ncfile_name = os.path.join(outloc, f'INS3D_L3_CF_4km_Ver02_D{today_date}.nc')
    try:
        ncfile = netcdf.Dataset(ncfile_name, 'w', format='NETCDF4')
        
        # Define dimensions - use actual data dimensions
        ncfile.createDimension('latitude', m)   # number of rows
        ncfile.createDimension('longitude', n)  # number of columns
        
        # Create coordinate variables
        lat_var = ncfile.createVariable('latitude', 'f8', ('latitude',))
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var.limits = '-10N to 45.5N in degrees'
        lat_var.comment = 'Positive latitude is North latitude, negative latitude is South latitude.'
        
        lon_var = ncfile.createVariable('longitude', 'f8', ('longitude',))
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var.limits = '44.5E to 105.5E in degrees'
        lon_var.comment = 'East longitude relative to Greenwich meridian.'
        
        # Fill coordinate variables
        # If the coordinate arrays don't match the data dimensions exactly,
        # create linearly spaced arrays based on the geographic bounds
        if len(latitude) != m:
            lat_var[:] = np.linspace(latitude.min(), latitude.max(), m)
        else:
            lat_var[:] = latitude
            
        if len(longitude) != n:
            lon_var[:] = np.linspace(longitude.min(), longitude.max(), n)
        else:
            lon_var[:] = longitude
        
        # Set global attributes
        ncfile.Title = 'INSAT-3D Cloud Mask'
        ncfile.Version = '2.0'
        ncfile.Projection = 'Geographic projection (WGS 84)'
        ncfile.Organization_Name = 'ISRO-DOS'
        ncfile.Processing_Centre = 'NRSC'
        ncfile.Satellite_Name = 'INSAT-3D'
        ncfile.Sensor = 'IMAGER'
        ncfile.Software = 'Python'
        ncfile.Resolution = '0.04 x 0.04 degrees'
        ncfile.Created_By = 'LAPD/ECSA'
        ncfile.Date_Processed = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add variables for each time point
        for i in range(len(fnD)):
            if i >= len(t):
                print(f'\t Warning: Skipping time index {i} (out of range)')
                continue
                
            time1 = int(t[i])
            
            # Format time string based on value
            if time1 >= 1000:
                time2 = f"{time1}"
            elif time1 >= 100:
                time2 = f"0{time1}"
            elif time1 >= 10:
                time2 = f"00{time1}"
            else:
                time2 = f"000{time1}"
            
            varname1 = f'cloud_mask_{time2}'
            varname2 = f'flag_{time2}'
            
            try:
                # Create cloud mask variable
                # Use the correct dimension order: (latitude, longitude)
                var_cm = ncfile.createVariable(varname1, 'f4', ('latitude', 'longitude'))
                var_cm.standard_name = 'cloud_mask'
                var_cm.long_name = 'Cloud Mask'
                var_cm.units = 'binary'
                var_cm.FLAG_1 = 'High level thick clouds'
                var_cm.FLAG_2 = 'Low level thick clouds'
                var_cm.FLAG_3 = 'Semi-transparent cirrus clouds'
                var_cm.FLAG_4 = 'Partial clouds'
                var_cm.valid_range = [0, 1]
                var_cm.missing_value = -999
                
                # Assign data without transposing since we're using (latitude, longitude) order
                var_cm[:, :] = CM_4kmG[:, :, i]
                
                # Create flag variable
                var_flag = ncfile.createVariable(varname2, 'f4', ('latitude', 'longitude'))
                var_flag.standard_name = 'cloud_flag'
                var_flag.long_name = 'Cloud Flag'
                var_flag.units = 'binary'
                var_flag.valid_range = [0, 1]
                var_flag.missing_value = -999
                
                # Assign flag data
                var_flag[:, :] = CFlag[:, :, i]
                
                print(f'\t Added variables for time {time2}')
                
            except Exception as e:
                print(f'\t Error adding variables for time {time2}: {str(e)}')
        
        # Close NetCDF file
        ncfile.close()
        print(f'\n\t Successfully wrote NetCDF file: {ncfile_name}')
        
    except Exception as e:
        print(f'\n\t Error writing NetCDF file: {str(e)}')
        # Make sure to close the file if it was opened
        try:
            ncfile.close()
        except:
            pass
    
    # Create TIF files for visualization
    gs = 0.04  # Grid spacing
    dgs = gs / 2  # Half grid spacing
    
    # Load coastline data once for all plots
    try:
        coast_data = robust_loadmat('coast.mat')
        long = coast_data.get('long', [])
        lat = coast_data.get('lat', [])
        has_coastline = True
    except Exception as e:
        print(f"\t Could not load coastline data: {str(e)}")
        has_coastline = False
    
    # Create directory for temporary TIF files
    tif_dir = os.path.join(outloc, 'temp_tifs')
    os.makedirs(tif_dir, exist_ok=True)
    
    # Generate plots for each time point
    tif_files = []
    
    for i in range(len(fnm)):
        if i >= CM_4kmG.shape[2]:
            print(f"\t Warning: Skipping frame {i} (out of range)")
            continue
            
        Fname = fnm[i]
        nmp = f'CM{Fname[6:20]}'  # Output image name
        tnm = f'CM - {Fname[6:15]} {Fname[16:20]}UTC'  # Title
        output_path = os.path.join(tif_dir, nmp + '.tif')
        
        print(f"\n\t Creating plot {i+1}/{len(fnm)}: {os.path.basename(output_path)}")
        
        # Verify data
        if np.all(np.isnan(CM_4kmG[:, :, i])) or np.sum(CM_4kmG[:, :, i]) == 0:
            print("\t Warning: Empty or all-NaN data, skipping plot")
            continue
            
        try:
            plt.figure(figsize=(10, 12))
            
            # Create plot

            img_data = CM_4kmG[:, :, i]

            # Create coordinate grids that match the data dimensions
            x = np.arange(img_data.shape[1] + 1) * gs - dgs + 45  # longitude
            y = np.arange(img_data.shape[0] + 1) * gs - dgs - 10  # latitude

            img = plt.pcolormesh(x, y, img_data, shading='auto', vmin=0, vmax=1)
                        
            # Add coastline if available
            if has_coastline:
                plt.plot(long, lat, 'k', linewidth=1)
            
            # Set colormap
            colors = [(0.258824, 0.776471, 1), (1, 1, 1)]  # Blue for clear, white for cloudy
            cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors, N=2)
            # Use the colormap directly in pcolormesh
            img = plt.pcolormesh(x, y, img_data, shading='auto', vmin=0, vmax=1, cmap=cmap)
            cbar = plt.colorbar(img, label='Cloud Mask', ticks=[0.25, 0.75])
            cbar.set_ticklabels(['Clear', 'Cloudy'])
            
            # Set axis properties
            plt.axis([45, 105, -10, 45])
            plt.xticks(range(45, 106, 10))
            plt.yticks(range(-10, 46, 10))
            plt.grid(True)
            
            # Add minor grid
            ax = plt.gca()
            ax.minorticks_on()
            ax.set_xticks(range(45, 106, 5), minor=True)
            ax.set_yticks(range(-10, 46, 5), minor=True)
            ax.grid(which='minor', alpha=0.2)
            
            # Labels and title
            plt.ylabel('Latitude (°N)', fontsize=14)
            plt.xlabel('Longitude (°E)', fontsize=14)
            plt.title(tnm, fontsize=14)
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\t Successfully saved: {os.path.basename(output_path)}")
            tif_files.append(output_path)
            
        except Exception as e:
            print(f"\t Error creating plot: {str(e)}")
        finally:
            plt.close()
    
    # Create GIF animation from TIF files
    if tif_files:
        print('\n\t Creating GIF animation...')
        try:
            # Parse date components
            year = today_date[6:10]
            day = today_date[0:2]
            
            # Convert month abbreviation to number
            months = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06', 
                     'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
            
            month = months.get(today_date[3:6], '01')  # Default to '01' if month not found
            
            # Create GIF filename
            gif_filename = os.path.join(outloc, f'CM_4km_{year}{month}{day}.gif')
            
            # Sort TIF files by name to ensure correct sequence
            tif_files.sort()
            
            # Create GIF using imageio
            frames = []
            for tif_file in tif_files:
                try:
                    frames.append(imageio.imread(tif_file))
                except Exception as e:
                    print(f"\t Error reading frame from {tif_file}: {str(e)}")
            
            if frames:
                imageio.mimsave(gif_filename, frames, duration=0.25, loop=0)
                print(f'\n\t GIF animation saved to: {gif_filename}')
            else:
                print('\n\t Error: No valid frames for GIF animation')
                
            # Create log file
            log_filename = os.path.join(outloc, f'INS3D_L3_CF_4km_Ver02_D{today_date}.txt')
            with open(log_filename, 'w') as f:
                f.write(f"Processing completed for {today_date}\n")
                f.write(f"Processed {len(tif_files)} valid frames\n")
                f.write(f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Remove temporary TIF files
            for tif_file in tif_files:
                try:
                    os.remove(tif_file)
                except Exception as e:
                    print(f"\t Error removing temporary file {tif_file}: {str(e)}")
            
            # Remove temporary directory if empty
            try:
                os.rmdir(tif_dir)
            except:
                pass
                
        except Exception as e:
            print(f"\n\t Error creating GIF animation: {str(e)}")
    else:
        print('\n\t No valid TIF files found for GIF animation')
    
    print("\n\t Processing complete!")

if __name__ == "__main__":
    main()