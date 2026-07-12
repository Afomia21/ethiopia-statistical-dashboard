import argparse
import io
import json
import zipfile
import xml.etree.ElementTree as ET

import requests

DDI_URL = "https://microdata.worldbank.org/metadata/export/6161/ddi"
OUT_PATH = "codebook.json"


def fetch_ddi_xml(url: str) -> bytes:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    content = r.content
    # The export may come back as a zip containing the actual XML file
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            xml_names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if not xml_names:
                raise RuntimeError(f"No XML found inside DDI zip. Contents: {z.namelist()}")
            return z.read(xml_names[0])
    return content


def strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def build_codebook(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)

    # Map internal file IDs (e.g. "F4") -> real file names (e.g. "sect3_hh_w5")
    file_names = {}
    for el in root.iter():
        if strip_ns(el.tag) == "fileDscr":
            file_id = el.get("ID") or el.get("id")
            for child in el.iter():
                if strip_ns(child.tag) == "fileName" and child.text:
                    if file_id:
                        file_names[file_id] = child.text.strip()
                    break

    codebook = {}
    for el in root.iter():
        if strip_ns(el.tag) != "var":
            continue
        varname = el.get("name")
        if not varname:
            continue

        label = None
        for child in el:
            if strip_ns(child.tag) == "labl":
                label = (child.text or "").strip()
                break
        if not label:
            continue

        files_attr = el.get("files") or el.get("file-id") or ""
        target_files = [file_names.get(fid, fid) for fid in files_attr.split()]
        if not target_files:
            target_files = ["_global"]

        for fname in target_files:
            codebook.setdefault(fname, {})[varname] = label

    return codebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="dump raw XML tag structure sample")
    args = parser.parse_args()

    print("Downloading DDI export...")
    xml_bytes = fetch_ddi_xml(DDI_URL)
    print(f"Downloaded {len(xml_bytes)} bytes.")

    if args.debug:
        root = ET.fromstring(xml_bytes)
        seen = set()
        for el in root.iter():
            tag = strip_ns(el.tag)
            if tag not in seen:
                seen.add(tag)
                print(tag, dict(el.attrib))
        return

    print("Parsing variables...")
    codebook = build_codebook(xml_bytes)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(codebook, f, indent=2, ensure_ascii=False)

    n_files = len(codebook)
    n_vars = sum(len(v) for v in codebook.values())
    print(f"Saved {OUT_PATH}: {n_files} files, {n_vars} variable labels total")

    # quick sanity check
    sample_file = next(iter(codebook), None)
    if sample_file:
        print(f"\nSample from '{sample_file}':")
        for k, v in list(codebook[sample_file].items())[:5]:
            print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()