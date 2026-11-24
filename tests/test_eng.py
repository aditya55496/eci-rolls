"""English roll parsing (engpipe 2026-11-25): line roles with wrap lines, label / gender snapping, EPIC repair, hint label kinds."""
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, 'eciocr', 'ocr'), os.path.join(R, 'eciocr', 'prod')]
import parse_eng as P  # noqa: E402


def test_roles_wrap_and_stamp():
    L = {'line0': 'Name : Smita Shetgaonkar', 'line1': 'Husbands Name: Ramdas Keshav', 'line2': 'Shetgaonkar',
         'line3': 'House Number : 968', 'line4': 'Age : 42 Gender : Female', 'line5': 'ADJU'}
    roles = P.box_roles([L, L], 6)
    assert roles == ['name', 'rel', None, 'house', 'age', 'drop']
    r = P.from_lines_eng(P.canonical(dict(L, serial='159', epic='TRW0302117'), roles))
    assert (r['name'], r['relation_label'], r['relation_name'], r['house_no'], r['age'], r['gender']) == (
        'Smita Shetgaonkar', 'Husbands Name', 'Ramdas Keshav Shetgaonkar', '968', '42', 'Female')


def test_label_missing_filled_by_position():
    good = {'line0': 'Name : A B', 'line1': "Father's Name: C D", 'line2': 'House Number : 1-13/A', 'line3': 'Age : 22 Gender : Male'}
    bad = {'line0': 'Name : A B', 'line1': "Fathcr'5 Nanne C D", 'line2': 'House Number : 1-13/A', 'line3': 'Age : 22 Gender : Male'}
    assert P.box_roles([good, bad, bad], 4) == ['name', 'rel', 'house', 'age']


def test_cleanup():
    r = P.from_lines_eng({'serial': '183', 'epic': 'YKS15/0423', 'line0': 'Name : ? SUJATA DODD!',
                          'line1': 'Fathers Name: MOHANA RAO DODD|', 'line2': 'House Number : 1-133', 'line3': 'Age : 32 Gender : Ma1e'})
    assert r['epic'] == 'YKS1570423' and r['name'] == 'SUJATA DODDI' and r['relation_name'] == 'MOHANA RAO DODDI'
    assert r['gender'] == 'Male' and r['relation_label'] == 'Fathers Name'
    assert P.fix_epic('AP/03/012/123456') == 'AP/03/012/123456'


def test_rel_canon():
    assert P.rel_canon("Father's Name") == P.rel_canon('Fathers Name') == P.rel_canon('Father Name') == 'father'
    assert P.rel_canon('Others') == P.rel_canon('Other') == 'other'
