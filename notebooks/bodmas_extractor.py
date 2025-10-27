#!/usr/bin/python
"""
Modernized EMBER/BODMAS feature extractor using LIEF ≥ 0.17
Compatible with modern NumPy / scikit-learn / Python 3.10+

Produces the 2 ,381-dimensional static feature vector used in
the BODMAS dataset.
"""

import re
import os
import json
import hashlib
import numpy as np
import lief
from sklearn.feature_extraction import FeatureHasher


# -----------------------------------------------------------
# Compatibility flags (for LIEF ≥ 0.17)
# -----------------------------------------------------------
LIEF_MAJOR, LIEF_MINOR, *_ = lief.__version__.split(".")
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or (int(LIEF_MAJOR) == 0 and int(LIEF_MINOR) >= 10)
LIEF_HAS_SIGNATURE = int(LIEF_MAJOR) > 0 or (int(LIEF_MAJOR) == 0 and int(LIEF_MINOR) >= 11)


# ===========================================================
#  Generic feature type base
# ===========================================================
class FeatureType(object):
    name = ""
    dim = 0

    def __repr__(self):
        return f"{self.name}({self.dim})"

    def raw_features(self, bytez, lief_binary):
        raise NotImplementedError

    def process_raw_features(self, raw_obj):
        raise NotImplementedError

    def feature_vector(self, bytez, lief_binary):
        return self.process_raw_features(self.raw_features(bytez, lief_binary))


# ===========================================================
#  Individual feature groups
# ===========================================================
class ByteHistogram(FeatureType):
    name, dim = "histogram", 256

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        s = counts.sum()
        return counts / s if s > 0 else counts


class ByteEntropyHistogram(FeatureType):
    name, dim = "byteentropy", 256

    def __init__(self, step=1024, window=2048):
        self.window, self.step = window, step

    def _entropy_bin_counts(self, block):
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2
        Hbin = min(int(H * 2), 15)
        return Hbin, c

    def raw_features(self, bytez, lief_binary):
        output = np.zeros((16, 16), dtype=int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if len(a) < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c
        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        arr = np.array(raw_obj, dtype=np.float32)
        s = arr.sum()
        return arr / s if s > 0 else arr


class SectionInfo(FeatureType):
    name, dim = "section", 5 + 50 * 5

    @staticmethod
    def _props(s):
        return [str(c).split(".")[-1] for c in s.characteristics_lists]

    def raw_features(self, bytez, lief_binary):
        if not lief_binary:
            return {"entry": "", "sections": []}
        try:
            section = lief_binary.section_from_rva(lief_binary.entrypoint - lief_binary.imagebase)
            entry_section = section.name if section else ""
        except Exception:
            entry_section = ""
        return {
            "entry": entry_section,
            "sections": [
                {
                    "name": s.name,
                    "size": s.size,
                    "entropy": getattr(s, "entropy", 0.0),
                    "vsize": s.virtual_size,
                    "props": self._props(s),
                }
                for s in lief_binary.sections
            ],
        }

    def process_raw_features(self, raw_obj):
        sections = raw_obj["sections"]
        general = [
            len(sections),
            sum(1 for s in sections if s["size"] == 0),
            sum(1 for s in sections if s["name"] == ""),
            sum(1 for s in sections if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"]),
            sum(1 for s in sections if "MEM_WRITE" in s["props"]),
        ]
        section_sizes = [(s["name"], s["size"]) for s in sections]
        section_entropy = [(s["name"], s["entropy"]) for s in sections]
        section_vsize = [(s["name"], s["vsize"]) for s in sections]
        entry = raw_obj["entry"]
        entry_props = [p for s in sections for p in s["props"] if s["name"] == entry]

        fh_pair = lambda n, data: FeatureHasher(n, input_type="pair").transform([data]).toarray()[0]
        fh_str = lambda n, data: FeatureHasher(n, input_type="string").transform([data]).toarray()[0]

        return np.hstack(
            [
                general,
                fh_pair(50, section_sizes),
                fh_pair(50, section_entropy),
                fh_pair(50, section_vsize),
                fh_str(50, [entry]),
                fh_str(50, list(entry_props)),
            ]
        ).astype(np.float32)


class ImportsInfo(FeatureType):
    name, dim = "imports", 1280

    def raw_features(self, bytez, lief_binary):
        imports = {}
        if lief_binary:
            for lib in lief_binary.imports:
                imports.setdefault(lib.name or "", [])
                for e in lib.entries:
                    val = f"ordinal{e.ordinal}" if e.is_ordinal else str(e.name)[:10000]
                    imports[lib.name].append(val)
        return imports

    def process_raw_features(self, raw_obj):
        libraries = list({l.lower() for l in raw_obj.keys()})
        imports = [f"{lib.lower()}:{e}" for lib, el in raw_obj.items() for e in el]
        fh = lambda n, data: FeatureHasher(n, input_type="string").transform([data]).toarray()[0]
        return np.hstack([fh(256, libraries), fh(1024, imports)]).astype(np.float32)


class ExportsInfo(FeatureType):
    name, dim = "exports", 128

    def raw_features(self, bytez, lief_binary):
        if not lief_binary:
            return []
        if LIEF_EXPORT_OBJECT:
            return [e.name[:10000] for e in lief_binary.exported_functions]
        return [str(e)[:10000] for e in lief_binary.exported_functions]

    def process_raw_features(self, raw_obj):
        return FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0].astype(np.float32)


class GeneralFileInfo(FeatureType):
    name, dim = "general", 10

    def raw_features(self, bytez, lief_binary):
        if not lief_binary:
            return {k: 0 for k in
                    ["size","vsize","has_debug","exports","imports","has_relocations","has_resources","has_signature","has_tls","symbols"]}
        return {
            "size": len(bytez),
            "vsize": lief_binary.virtual_size,
            "has_debug": int(lief_binary.has_debug),
            "exports": len(lief_binary.exported_functions),
            "imports": len(lief_binary.imported_functions),
            "has_relocations": int(lief_binary.has_relocations),
            "has_resources": int(lief_binary.has_resources),
            "has_signature": int(getattr(lief_binary, "has_signatures", getattr(lief_binary, "has_signature", 0))),
            "has_tls": int(lief_binary.has_tls),
            "symbols": len(lief_binary.symbols),
        }

    def process_raw_features(self, r):
        return np.array(
            [r[k] for k in
             ["size","vsize","has_debug","exports","imports","has_relocations","has_resources","has_signature","has_tls","symbols"]],
            dtype=np.float32)


class HeaderFileInfo(FeatureType):
    name, dim = "header", 62

    def raw_features(self, bytez, lief_binary):
        if not lief_binary:
            return {"coff": {}, "optional": {}}
        hdr = lief_binary.header
        opt = lief_binary.optional_header
        return {
            "coff": {
                "timestamp": hdr.time_date_stamps,
                "machine": str(hdr.machine).split(".")[-1],
                "characteristics": [str(c).split(".")[-1] for c in hdr.characteristics_list],
            },
            "optional": {
                "subsystem": str(opt.subsystem).split(".")[-1],
                "dll_characteristics": [str(c).split(".")[-1] for c in opt.dll_characteristics_lists],
                "magic": str(opt.magic).split(".")[-1],
                "major_image_version": opt.major_image_version,
                "minor_image_version": opt.minor_image_version,
                "major_linker_version": opt.major_linker_version,
                "minor_linker_version": opt.minor_linker_version,
                "major_operating_system_version": opt.major_operating_system_version,
                "minor_operating_system_version": opt.minor_operating_system_version,
                "major_subsystem_version": opt.major_subsystem_version,
                "minor_subsystem_version": opt.minor_subsystem_version,
                "sizeof_code": opt.sizeof_code,
                "sizeof_headers": opt.sizeof_headers,
                "sizeof_heap_commit": opt.sizeof_heap_commit,
            },
        }

    def process_raw_features(self, r):
        fh = lambda n, data: FeatureHasher(n, input_type="string").transform([data]).toarray()[0]
        return np.hstack([
            r["coff"]["timestamp"],
            fh(10, [r["coff"]["machine"]]),
            fh(10, list(r["coff"]["characteristics"])),
            fh(10, [r["optional"]["subsystem"]]),
            fh(10, list(r["optional"]["dll_characteristics"])),
            fh(10, [r["optional"]["magic"]]),
            r["optional"]["major_image_version"],
            r["optional"]["minor_image_version"],
            r["optional"]["major_linker_version"],
            r["optional"]["minor_linker_version"],
            r["optional"]["major_operating_system_version"],
            r["optional"]["minor_operating_system_version"],
            r["optional"]["major_subsystem_version"],
            r["optional"]["minor_subsystem_version"],
            r["optional"]["sizeof_code"],
            r["optional"]["sizeof_headers"],
            r["optional"]["sizeof_heap_commit"],
        ]).astype(np.float32)


class StringExtractor(FeatureType):
    name, dim = "strings", 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def __init__(self):
        self._all = re.compile(b"[\x20-\x7f]{5,}")
        self._paths = re.compile(b"c:\\\\", re.I)
        self._urls = re.compile(b"https?://", re.I)
        self._reg = re.compile(b"HKEY_")
        self._mz = re.compile(b"MZ")

    def raw_features(self, bytez, lief_binary):
        s = self._all.findall(bytez)
        if not s:
            return {"numstrings": 0, "avlength": 0, "printabledist": [0]*96,
                    "printables": 0, "entropy": 0, "paths": 0, "urls": 0, "registry": 0, "MZ": 0}
        lens = [len(x) for x in s]
        avlen = np.mean(lens)
        as_shift = [b - 0x20 for b in b"".join(s)]
        c = np.bincount(as_shift, minlength=96)
        p = c / max(1, c.sum())
        wh = np.where(c)[0]
        H = float(np.sum(-p[wh]*np.log2(p[wh]))) if len(wh) else 0.0
        return {"numstrings": len(s), "avlength": avlen, "printabledist": c.tolist(),
                "printables": int(c.sum()), "entropy": H,
                "paths": len(self._paths.findall(bytez)), "urls": len(self._urls.findall(bytez)),
                "registry": len(self._reg.findall(bytez)), "MZ": len(self._mz.findall(bytez))}

    def process_raw_features(self, r):
        div = float(r["printables"]) if r["printables"] > 0 else 1.0
        return np.hstack([
            r["numstrings"], r["avlength"], r["printables"],
            np.array(r["printabledist"], dtype=np.float32)/div,
            r["entropy"], r["paths"], r["urls"], r["registry"], r["MZ"]
        ]).astype(np.float32)


class DataDirectories(FeatureType):
    name, dim = "datadirectories", 30

    def raw_features(self, bytez, lief_binary):
        if not lief_binary:
            return []
        return [{"size": d.size, "virtual_address": d.rva} for d in lief_binary.data_directories]

    def process_raw_features(self, r):
        out = np.zeros(30, dtype=np.float32)
        for i, d in enumerate(r[:15]):
            out[2*i], out[2*i+1] = d["size"], d["virtual_address"]
        return out


# ===========================================================
#  Extractor orchestrator
# ===========================================================
class PEFeatureExtractor:
    def __init__(self, feature_version=2):
        self.features = [
            ByteHistogram(), ByteEntropyHistogram(), StringExtractor(),
            GeneralFileInfo(), HeaderFileInfo(), SectionInfo(),
            ImportsInfo(), ExportsInfo(), DataDirectories()
        ]
        self.dim = sum(f.dim for f in self.features)

    def raw_features(self, bytez):
        try:
            lief_binary = lief.PE.parse(list(bytez))
        except Exception as e:
            print("LIEF parse error:", e)
            lief_binary = None
        features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features.update({f.name: f.raw_features(bytez, lief_binary) for f in self.features})
        return features

    def process_raw_features(self, r):
        return np.hstack([f.process_raw_features(r[f.name]) for f in self.features]).astype(np.float32)

    def feature_vector(self, bytez):
        return self.process_raw_features(self.raw_features(bytez))


# ===========================================================
#  Helper function for AIMaP app
# ===========================================================
def extract_2381_from_exe(path):
    """Return the 2 ,381-dimensional BODMAS feature vector for one .exe."""
    with open(path, "rb") as f:
        data = f.read()
    extractor = PEFeatureExtractor(feature_version=2)
    return extractor.feature_vector(data)





# ===========================================================
#  Helper function for AIMaP app
# ===========================================================
def extract_2381_from_exe(path):
    """Return the 2 ,381-dimensional BODMAS feature vector for one .exe."""
    with open(path, "rb") as f:
        data = f.read()
    extractor = PEFeatureExtractor(feature_version=2)
    return extractor.feature_vector(data)
