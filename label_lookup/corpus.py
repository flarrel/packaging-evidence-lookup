"""Build a small packaging corpus from DailyMed label records and label art."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from uuid import UUID
from xml.etree import ElementTree

from PIL import UnidentifiedImageError

from .downloads import download
from .images import load_image

ROOT = "https://dailymed.nlm.nih.gov/dailymed"
API = f"{ROOT}/services/v2"
BRANDS = (
    "Advil", "Afrin", "Aleve", "Allegra", "Aquaphor", "Azo", "Benadryl", "Burt's Bees",
    "Carmex", "Claritin", "Colace", "Cortizone", "Crest", "Culturelle", "Delsym",
    "Dramamine", "Dulcolax", "Eucerin", "Excedrin", "Flonase", "Gold Bond",
    "Hydroxycut", "Listerine", "Mucinex", "Neosporin", "Nizoral", "Orajel",
    "Pepcid", "Pepto-Bismol", "Preparation H", "Prilosec", "Rolaids", "Sensodyne",
    "Sudafed", "Theraflu", "Tums", "Tylenol", "Vaseline", "Visine", "Zyrtec",
)


@dataclass(frozen=True)
class Record:
    id: str
    family: str
    name: str
    codes: tuple[str, ...]
    package: tuple[str, ...]
    source: str
    image_source: str
    image: str
    image_sha256: str
    version: int

    @property
    def text(self) -> str:
        return " ".join((self.name, *self.codes, *self.package))

    def evidence(self, field: str) -> str:
        return {
            "name": self.name,
            "codes": ", ".join(self.codes),
            "package": "; ".join(self.package),
        }[field]


def read_json(url: str, cache: Path):
    return json.loads(download(url, cache, 3_000_000).read_text())["data"]


def read_label(work: Path, brand: str, row: dict) -> Record | None:
    setid = str(UUID(row["setid"]))
    raw = work / "data" / setid
    packaging = read_json(f"{API}/spls/{setid}/packaging.json", raw / "packaging.json")
    media = read_json(f"{API}/spls/{setid}/media.json", raw / "media.json")
    xml = download(f"{API}/spls/{setid}.xml", raw / "label.xml", 5_000_000)
    root = ElementTree.parse(xml).getroot()
    namespace = {"hl7": "urn:hl7-org:v3"}
    names = set()
    for section in root.findall(".//hl7:section", namespace):
        code = section.find("hl7:code", namespace)
        if code is None or code.get("code") != "51945-4":
            continue
        names.update(
            node.get("value") for node in section.findall(".//hl7:reference", namespace)
            if node.get("value")
        )

    candidates = [
        entry for entry in media["media"]
        if entry["name"] in names and entry["mime_type"].startswith("image/")
    ]
    products = packaging["products"]
    if (not candidates or len(products) != 1
            or packaging["spl_version"] != media["spl_version"]):
        return None
    product = products[0]
    codes = tuple(sorted({
        row["ndc"] for row in product["packaging"] if isinstance(row.get("ndc"), str) and row["ndc"]
    }))
    package = tuple(sorted({
        " ".join(description.split())
        for row in product["packaging"] for description in (row.get("package_descriptions") or [])
        if isinstance(description, str) and description.strip()
    }))
    if not product["product_name"].strip() or not codes or not package:
        return None

    candidate = candidates[0]
    url = candidate["url"]
    if urlsplit(url).netloc != "dailymed.nlm.nih.gov":
        return None
    image_path = download(url, raw / "package-image", 4_000_000)
    image = load_image(image_path)
    if min(image.size) < 64:
        return None
    with image_path.open("rb") as stream:
        image_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return Record(
        id=setid, family=brand, name=product["product_name"], codes=codes, package=package,
        source=f"{ROOT}/drugInfo.cfm?setid={setid}", image_source=url,
        image=str(image_path.relative_to(work)), image_sha256=image_digest,
        version=packaging["spl_version"],
    )


def prepare(work: Path) -> dict:
    def collect(brand: str) -> tuple[list[Record], list[str]]:
        results, errors = [], []
        query = urlencode({"drug_name": brand, "name_type": "brand", "pagesize": 6})
        key = hashlib.sha256(brand.encode()).hexdigest()[:16]
        try:
            rows = read_json(f"{API}/spls.json?{query}", work / "data/search" / f"{key}.json")
        except (URLError, TimeoutError, ValueError) as error:
            return results, [f"{brand}: {error}"]
        for row in rows:
            try:
                record = read_label(work, brand, row)
                if record:
                    results.append(record)
                else:
                    errors.append(f"{brand}/{row['setid']}: missing usable single-product label art")
            except (URLError, TimeoutError, UnidentifiedImageError, ValueError, ElementTree.ParseError) as error:
                errors.append(f"{brand}/{row['setid']}: {error}")
            if len(results) == 2:
                break
        return results, errors

    unique, digests, errors = {}, set(), []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for records, failures in executor.map(collect, BRANDS):
            errors.extend(failures)
            for record in records:
                if record.id in unique or record.image_sha256 in digests:
                    errors.append(f"{record.id}: duplicate record or exact image")
                    continue
                unique[record.id] = record
                digests.add(record.image_sha256)

    if len({record.family for record in unique.values()}) < 12:
        raise RuntimeError(f"Need 12 usable product families. Collected {len(unique)} records. {errors[:6]}")
    records = tuple(unique.values())
    splits = split_families(records)
    payload = {
        "source": ROOT, "records": [record.__dict__ for record in records],
        "splits": splits, "rejected": errors,
    }
    (work / "corpus.json").write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "records": len(records), "rejected": len(errors),
        "families": {key: len(value) for key, value in splits.items()},
        "manifest": str(work / "corpus.json"),
    }


def split_families(records: tuple[Record, ...]) -> dict[str, list[str]]:
    families = sorted(
        {r.family for r in records},
        key=lambda item: hashlib.sha256(f"packaging-v1:{item}".encode()).digest(),
    )
    pool = max(2, len(families) // 5)
    return {"train": families[2 * pool:], "validation": families[pool:2 * pool], "test": families[:pool]}


def load(work: Path) -> tuple[list[Record], dict[str, list[str]]]:
    manifest = json.loads((work / "corpus.json").read_text())
    return [Record(**row) for row in manifest["records"]], manifest["splits"]
