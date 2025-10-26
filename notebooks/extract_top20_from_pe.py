# extract_top20_from_pe.py
import lief
import os
import numpy as np
import hashlib
import re

# Make sure TOP20_NAMES match the order used in training
TOP20_NAMES = [
 "hash_coff_machine_type",
 "data_dir_virtual_size_3",
 "data_dir_size_4",
 "data_dir_virtual_size_1",
 "hash_optional_dll_characteristics",
 "hash_optional_subsystem",
 "minor_operating_system_version",
 "occurrences_c_colon",
 "num_sections_empty_name",
 "data_dir_size_6",
 "hash_lib_func_346",
 "bytehist_94",
 "data_dir_size_1",
 "num_symbols",
 "bytehist_31",
 "hash_lib_func_495",
 "hash_entrypoint_props_36",
 "bytehist_254",
 "data_dir_virtual_size_11",
 "freq_printable_char_94"
]

def sha1_bucket_int(s, modulus=2**31-1):
    """Return stable integer from bytes/string via sha1, mapped to int range."""
    if isinstance(s, str):
        s = s.encode('utf8', errors='ignore')
    h = hashlib.sha1(s).hexdigest()
    return int(h[:15], 16) % modulus

def entropy_bytes(b: bytes):
    if not b:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def extract_strings_from_bytes(b: bytes, min_len=4):
    # crude ascii string extractor
    try:
        s = b.decode('latin1', errors='ignore')
    except:
        return []
    return re.findall(r'[\x20-\x7E]{%d,}' % min_len, s)

def count_byte_value(b: bytes, value:int):
    if not b:
        return 0
    arr = np.frombuffer(b, dtype=np.uint8)
    return int((arr == value).sum())

def extract_top20_from_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    binary = lief.parse(path)
    if binary is None:
        raise ValueError("Not a valid PE file or LIEF failed to parse")
    # Whole file bytes
    with open(path, "rb") as f:
        full_bytes = f.read()
    # 1. hash COFF machine type - raw integer hashed
    machine = getattr(binary.header, "machine", 0)
    v1 = sha1_bucket_int(str(machine))
    # 2-? Data directories - map to optional_header.data_directories
    dd = {}
    try:
        for i, d in enumerate(binary.optional_header.data_directories):
            # some entries may not have size/virtual_size attributes depending on lief version
            size = getattr(d, "size", 0)
            vsize = getattr(d, "virtual_size", 0) if hasattr(d, "virtual_size") else 0
            dd[i] = {"size": int(size), "virtual_size": int(vsize)}
    except Exception:
        # fallback zero
        dd = {i: {"size":0, "virtual_size":0} for i in range(16)}
    # 2. data_dir_virtual_size_3  (index 3 virtual size)
    v2 = dd.get(3, {}).get("virtual_size", 0)
    # 3. data_dir_size_4
    v3 = dd.get(4, {}).get("size", 0)
    # 4. data_dir_virtual_size_1
    v4 = dd.get(1, {}).get("virtual_size", 0)
    # 5. hash optional dll_characteristics
    dll_chars = getattr(binary.optional_header, "dll_characteristics", 0)
    v5 = sha1_bucket_int(str(dll_chars))
    # 6. hash optional subsystem
    subsystem = getattr(binary.optional_header, "subsystem", 0)
    v6 = sha1_bucket_int(str(subsystem))
    # 7. minor_operating_system_version
    minor_os = getattr(binary.optional_header, "minor_operating_system", 0)
    v7 = int(minor_os)
    # 8. occurrences of "c:\" (ignore case) in strings
    strings = extract_strings_from_bytes(full_bytes)
    occurrences_c = 0
    for s in strings:
        occurrences_c += s.lower().count("c:\\")
    v8 = int(occurrences_c)
    # 9. # of sections with empty name
    empty_name_count = sum(1 for s in binary.sections if (not s.name) or (s.name.strip() == ""))
    v9 = int(empty_name_count)
    # 10. data_dir_size_6
    v10 = dd.get(6, {}).get("size", 0)
    # 11. hash on list of library:function - 346  (we'll hash sorted "lib:func" list)
    imports = []
    try:
        for lib in binary.imports:
            libname = getattr(lib, "name", "") or ""
            for e in lib.entries:
                funcname = getattr(e, "name", "") or str(getattr(e, "ordinal", ""))
                imports.append(f"{libname.lower()}:{funcname.lower()}")
    except:
        imports = []
    if imports:
        imports_sorted = sorted(set(imports))
        imports_join = "|".join(imports_sorted)
        v11 = sha1_bucket_int(imports_join)
    else:
        v11 = 0
    # 12. ByteHistogram-94  (count freq of byte value 94 (caret '^') or the histogram bucket used)
    # earlier you had ByteHistogram-94; I'll interpret as byte value 94 frequency
    v12 = count_byte_value(full_bytes, 94)
    # 13. data_dir_size_1
    v13 = dd.get(1, {}).get("size", 0)
    # 14. # of Symbols - try header.coff.numberof_symbols or 0
    try:
        num_symbols = int(getattr(binary, "numberof_symbols", 0))
    except:
        # some LIEF versions expose header.numberof_symbols
        num_symbols = int(getattr(binary.header, "numberof_symbols", 0) if hasattr(binary, "header") else 0)
    v14 = num_symbols
    # 15. ByteHistogram-31
    v15 = count_byte_value(full_bytes, 31)
    # 16. hash on list of library:function - 495 (we will reuse imports hash but different bucket)
    v16 = sha1_bucket_int(imports_join if imports else "")
    # 17. hash on list of properties of entry point - 36
    try:
        ep = getattr(binary.optional_header, "addressof_entrypoint", 0)
    except:
        ep = 0
    ep_props = f"{ep}_{binary.entrypoint if hasattr(binary,'entrypoint') else ''}"
    v17 = sha1_bucket_int(ep_props)
    # 18. ByteHistogram-254
    v18 = count_byte_value(full_bytes, 254)
    # 19. data_dir_virtual_size_11
    v19 = dd.get(11, {}).get("virtual_size", 0)
    # 20. freq of printable characters - '^' (interpret as frequency of caret '^' i.e., 94)
    total_bytes = max(1, len(full_bytes))
    freq_printable_94 = v12 / total_bytes
    v20 = float(freq_printable_94)

    feature_vector = [
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20
    ]
    return np.array(feature_vector, dtype=float)
