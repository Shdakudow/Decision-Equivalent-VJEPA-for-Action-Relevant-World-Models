"""Verify the published manuscript files without training or network access."""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    metadata = json.loads((ROOT / 'paper/manuscript_metadata.json').read_text())
    pdf_path = ROOT / metadata['manuscript']['path']
    source_path = ROOT / metadata['source_bundle']['path']
    assert digest(pdf_path.read_bytes()) == metadata['manuscript']['sha256']
    assert digest(source_path.read_bytes()) == metadata['source_bundle']['sha256']
    reader = PdfReader(pdf_path)
    assert reader.metadata['/Title'] == metadata['title']
    assert len(reader.pages) == metadata['total_pdf_pages']
    refs = next(i + 1 for i, p in enumerate(reader.pages)
                if re.search(r'^References$', p.extract_text(), re.M))
    assert refs == metadata['references_start_page']
    assert refs - 1 == metadata['main_content_pages'] <= 12
    assert reader.metadata.get('/Author', '') == ''
    assert 'Anonymous authors' in reader.pages[0].extract_text()

    with zipfile.ZipFile(source_path) as archive:
        assert archive.testzip() is None
        assert 'main.tex' in archive.namelist()
        assert 'implementation/audit_training_path.py' in archive.namelist()
        source_title = re.search(r'pdftitle=\{([^}]+)\}',
                                 archive.read('main.tex').decode())[1]
        assert source_title == metadata['title']
        result_files = sorted((ROOT / 'results').glob('*.csv'))
        assert len(result_files) == metadata['original_aggregate_result_CSVs'] == 53
        for path in result_files:
            assert archive.read('evidence/results/' + path.name) == path.read_bytes(), path
        rows = list(csv.DictReader(io.StringIO(
            archive.read('implementation/RELEASE_CHECKSUMS.csv').decode())))
        with (ROOT / 'paper/released_implementation_checksums.csv').open() as handle:
            assert list(csv.DictReader(handle)) == rows
        for row in rows:
            payload = (ROOT / row['path']).read_bytes()
            assert digest(payload) == row['sha256'], row['path']
            assert archive.read('implementation/' + row['path']) == payload, row['path']

    print(json.dumps({
        'paper_and_source_hashes_verified': True,
        'title': metadata['title'],
        'main_content_pages': refs - 1,
        'references_start_page': refs,
        'total_pdf_pages': len(reader.pages),
        'anonymous_review_PDF': True,
        'original_result_CSVs_verified': len(result_files),
        'released_implementation_files_verified': len(rows),
        'historical_run_execution_version_verified': False,
        'new_experiments_run': False,
    }, indent=2))


if __name__ == '__main__':
    main()
