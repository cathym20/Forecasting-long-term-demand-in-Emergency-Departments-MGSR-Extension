"""Build the Sub-ICB monthly panel used by the extension analysis.

Target: emergency admissions (Non-Elective) per Sub-ICB per month, coded to the patient's
home Sub-ICB in the HES Monthly Activity Report. Features: GP appointments (Sub-ICB native),
NHS 111 calls (contract-level, stratified to Sub-ICB by population share), Fingertips QOF
prevalence indicators (ICB-level, replicated to Sub-ICBs via the parent ICB), ONS mid-year
population and age structure."""

import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


HES_FILES = [
    '../data/hes/HES_MAR_2022-23_M13.csv',
    '../data/hes/HES_MAR_2023-24_M12.csv',
    '../data/hes/HES_MAR_2024-25_M13.csv',
    '../data/hes/HES_MAR_2025-26_M13.csv',]
POPULATION_FILE = '../data/population/sapeicb20222024.xlsx'
GP_FOLDER = '../data/gp/'
IUC_FOLDER = '../data/iuc/'
AMBSYS_FILE = '../data/ambulance/AmbSYS-Time-Series-to-20260531-k58VJ.xlsx'
FINGERTIPS_FILE = '../new_data/fingertips_icb_features.csv'
OUTPUT_FILE = '../new_data/master_hes_2022_2025.csv'

MONTH_MAP = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
             'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
MONTH_ABBR = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
              7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}


def load_hes(filepaths):
    """Read the HES MAR CSVs and keep Non-Elective (emergency admission) counts plus the
    four service-activity features used by the extended Level-0a block. Sub-ICB rows are
    flagged with Organisation breakdown == 'CCG' (legacy column name kept post-reorganisation).
    The ICB name is reconstructed by stripping the trailing region suffix off the Sub-ICB name."""
    # HES source columns -> internal names used in the master panel
    col_map = {
        'All specialties: Non-Elective':                          'emergency_admissions',
        'All specialties: Elective total':                        'elective_total',
        'All specialties: First attendances seen - GP referrals': 'gp_referrals',
        'All specialties: First attendances DNA':                 'outpatient_dna',}

    frames = []
    for fp in filepaths:
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, low_memory=False)
        ccg = df[df['Organisation breakdown'] == 'CCG'].copy()

        # suppressed values come through as '*'
        for src, dst in col_map.items():
            if src in ccg.columns:
                ccg[dst] = pd.to_numeric(ccg[src].replace('*', np.nan), errors='coerce')
            else:
                ccg[dst] = np.nan

        # first attendance total = GP referrals + other referrals (both "seen")
        gp_ref = pd.to_numeric(
            ccg.get('All specialties: First attendances seen - GP referrals',
                    pd.Series(np.nan, index=ccg.index)).replace('*', np.nan),
            errors='coerce').fillna(0)
        other_ref = pd.to_numeric(
            ccg.get('All specialties: First attendances seen - other referrals',
                    pd.Series(np.nan, index=ccg.index)).replace('*', np.nan),
            errors='coerce').fillna(0)
        ccg['outpatient_first_total'] = gp_ref + other_ref

        parts = ccg['Activity Month'].str.split(' ', expand=True)
        ccg['month'] = parts[0]
        ccg['year'] = parts[1].astype(float)
        ccg = ccg.rename(columns={'Org Code': 'ccg', 'Org Name': 'org_name'})
        ccg['icb_name'] = ccg['org_name'].str.replace(r'\s*-\s*\w+$', '', regex=True).str.strip()

        keep_cols = ['ccg', 'icb_name', 'year', 'month',
                     'emergency_admissions', 'elective_total',
                     'outpatient_first_total', 'outpatient_dna', 'gp_referrals']
        frames.append(ccg[keep_cols])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined['icb_name'].str.contains('COMMISSIONING HUB', case=False, na=False)]
    combined = combined[combined['icb_name'] != '-']
    return combined


def load_population(filepath):
    """One sheet per year in the ONS Sub-ICB SAPE file. Trailing three chars of the row
    label are the NHS code. Sum age >=65 columns for pop_65plus and scale total population
    to units of 10,000 to match the paper's convention.

    Note: skiprows=3 lands row 0 on the actual header row (SICBL Code, ..., Total, F0, F1,
    ..., F90, M0, ..., M90). skiprows=4 as previously used skipped the header and treated
    the first data row as the header, silently returning zero for pop_65plus."""
    xl = pd.ExcelFile(filepath)
    frames = []
    for sheet in xl.sheet_names:
        m = re.search(r'Mid[-\s]?(\d{4})', sheet)
        if not m:
            continue
        year = int(m.group(1))
        raw = pd.read_excel(filepath, sheet_name=sheet, header=None, skiprows=3)

        # Identify age columns once per sheet from the header row (F0..F90, M0..M90).
        header_row = raw.iloc[0]
        age_cols_65plus = []
        for c in range(7, raw.shape[1]):
            cn = str(header_row.get(c, ''))
            age_match = re.search(r'[FM](\d+)', cn)
            if age_match and int(age_match.group(1)) >= 65:
                age_cols_65plus.append(c)

        rows = []
        # data starts at row 1 (row 0 is the header)
        for _, r in raw.iloc[1:].iterrows():
            name = str(r[1])
            code_match = re.search(r'-\s*([0-9A-Z]{3})\s*$', name)
            if not code_match:
                continue
            code = code_match.group(1)
            total = pd.to_numeric(r[6], errors='coerce')
            if pd.isna(total):
                continue
            pop_65 = 0
            for c in age_cols_65plus:
                val = pd.to_numeric(r[c], errors='coerce')
                if not pd.isna(val):
                    pop_65 += val
            rows.append({
                'ccg': code, 'year': float(year),
                'population': total / 10000,
                'pop_65plus': pop_65,
                'pct_65plus': (pop_65 / total * 100) if total > 0 else np.nan,
            })
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def load_gp_data(folder):
    """Each monthly Appointments in General Practice release has a 'Table 4' sheet with
    counts by Sub-ICB. Filename gives the reporting month. Column 1 is the NHS code,
    column 4 is Total Count of Appointments."""
    if not os.path.exists(folder):
        return pd.DataFrame(columns=['ccg', 'year', 'month', 'gp_appt_available'])

    files = sorted([f for f in os.listdir(folder) if f.endswith(('.xlsx', '.xls'))])
    frames = []
    for f in files:
        try:
            path = os.path.join(folder, f)
            month_match = re.search(
                r'(January|February|March|April|May|June|July|August|September|October|November|December)'
                r'[\s_-]*(\d{4})', f, re.I)
            if not month_match:
                continue
            month_str = month_match.group(1).lower()
            file_year = int(month_match.group(2))

            xl = pd.ExcelFile(path)
            t4 = next((s for s in xl.sheet_names if s.strip().lower() in ('table 4', 'table4')), None)
            if t4 is None:
                continue

            df = pd.read_excel(path, sheet_name=t4, header=None)

            # Sub-ICB rows have 'Sub' in the label column; NHS code is a 3-char alphanumeric
            sub_mask = df[0].astype(str).str.contains('Sub', case=False, na=False)
            sub = df[sub_mask].copy()
            if len(sub) == 0:
                continue

            sub['ccg'] = sub[1].astype(str).str.strip()
            sub['count'] = pd.to_numeric(sub[4], errors='coerce')
            sub = sub[sub['ccg'].str.match(r'^[0-9A-Z]{3}$', na=False)]
            sub = sub[sub['count'].notna()]

            sub['month'] = MONTH_ABBR[MONTH_MAP[month_str]]
            sub['year'] = float(file_year)
            frames.append(sub[['ccg', 'year', 'month', 'count']].rename(
                columns={'count': 'gp_appt_available'}))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=['ccg', 'year', 'month', 'gp_appt_available'])

    return pd.concat(frames, ignore_index=True)


def load_iuc(folder):
    """Each IUC release contains a Raw data sheet with contract-month call volumes and a
    Current ICB (or ADC ICB) mapping sheet linking contracts to Sub-ICBs. Item numbers 1
    and 2 (or A01/A02 in older files) are calls offered and answered."""
    if not os.path.exists(folder):
        return pd.DataFrame(columns=['contract_code','year','month','calls_offered','calls_answered']), \
               pd.DataFrame(columns=['sub_icb','contract_code'])

    files = sorted([f for f in os.listdir(folder) if f.endswith('.xlsx')])
    frames = []
    for f in files:
        try:
            path = os.path.join(folder, f)
            xl = pd.ExcelFile(path)

            map_sheet = next((s for s in xl.sheet_names if 'current icb' in s.lower() or 'adc icb' in s.lower()), None)
            if map_sheet is None:
                continue

            mdf = pd.read_excel(path, sheet_name=map_sheet, header=None, skiprows=3)
            mapping = pd.DataFrame({
                'sub_icb': mdf[0].astype(str).str.strip(),
                'contract_code': mdf[3].astype(str).str.strip(),
            })
            mapping = mapping[mapping['sub_icb'].str.match(r'^[0-9A-Z]{3,5}$', na=False)]

            raw_sheet = 'Raw' if 'Raw' in xl.sheet_names else 'Raw Data'
            data = pd.read_excel(path, sheet_name=raw_sheet, header=0)
            data.columns = data.columns.str.strip()

            # column names drift between releases so match on substring
            col_map = {}
            for c in data.columns:
                cu = c.upper()
                if 'CONTRACT' in cu and 'CODE' in cu: col_map[c] = 'contract_code'
                elif 'ITEM' in cu and 'NUMBER' in cu: col_map[c] = 'ITEM_NUMBER'
                elif cu == 'VALUE': col_map[c] = 'VALUE'
            data = data.rename(columns=col_map)

            data['ITEM_NUMBER'] = data['ITEM_NUMBER'].astype(str).str.strip()
            data['ITEM_NUMBER'] = data['ITEM_NUMBER'].replace({'A01': '1', 'A02': '2'})
            data = data[data['ITEM_NUMBER'].isin(['1', '2'])]
            data['VALUE'] = pd.to_numeric(data['VALUE'], errors='coerce')

            if 'Date_deriv' in data.columns:
                data['date'] = pd.to_datetime(data['Date_deriv'], errors='coerce')
            elif 'REPORTING_PERIOD' in data.columns:
                data['date'] = pd.to_datetime(data['REPORTING_PERIOD'], format='%Y-%m-%d', errors='coerce')
            else:
                continue

            data['year'] = data['date'].dt.year.astype(float)
            data['month'] = data['date'].dt.month.map(MONTH_ABBR)

            pivot = data.pivot_table(index=['contract_code', 'year', 'month'],
                                     columns='ITEM_NUMBER', values='VALUE', aggfunc='sum').reset_index()
            pivot.columns = [str(c).strip() for c in pivot.columns]
            pivot = pivot.rename(columns={'1': 'calls_offered', '2': 'calls_answered'})
            frames.append((pivot, mapping))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=['contract_code','year','month','calls_offered','calls_answered']), \
               pd.DataFrame(columns=['sub_icb','contract_code'])

    all_data = pd.concat([f[0] for f in frames], ignore_index=True)
    all_maps = pd.concat([f[1] for f in frames], ignore_index=True).drop_duplicates(subset=['sub_icb','contract_code'])
    return all_data, all_maps


def stratify_iuc_to_subicb(iuc_raw, icb_map, population):
    """Split contract-level 111 counts across Sub-ICBs by population share. Per-capita
    rates derived from this end up identical within a contract, which is one of the
    discontinuities discussed in the dissertation."""
    c2s = icb_map.dropna(subset=['contract_code', 'sub_icb']).groupby('contract_code')['sub_icb'].apply(list).to_dict()
    records = []
    for _, row in iuc_raw.iterrows():
        subs = c2s.get(row['contract_code'], [])
        if not subs:
            continue
        pops = population[(population['ccg'].isin(subs)) & (population['year'] == row['year'])][['ccg', 'population']]
        total_pop = pops['population'].sum()
        if total_pop == 0:
            continue
        for _, p in pops.iterrows():
            share = p['population'] / total_pop
            records.append({
                'ccg': p['ccg'], 'year': row['year'], 'month': row['month'],
                '111_offered': (row.get('calls_offered', 0) or 0) * share,
                '111_answered': (row.get('calls_answered', 0) or 0) * share,
            })
    df = pd.DataFrame(records)
    if len(df):
        df = df.groupby(['ccg', 'year', 'month']).sum().reset_index()
    return df


def load_fingertips_subicb(filepath):
    """Load the ICB-level features produced by fingertips.py. Depression QOF has a
    one-year publication lag, so forward-fill within each ICB to cover missing later years."""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    ft = pd.read_csv(filepath)
    # obesity turned out too sparse across ICBs to use
    ft = ft.drop(columns=[c for c in ft.columns if 'obesity' in c.lower()], errors='ignore')
    if 'fx_depression__qof_prevalence' in ft.columns:
        ft = ft.sort_values(['icb_code', 'year'])
        ft['fx_depression__qof_prevalence'] = ft.groupby('icb_code')['fx_depression__qof_prevalence'].ffill()
    return ft


def build_master(hes, gp, iuc, population, fingertips=None):
    hes_cols = ['ccg', 'icb_name', 'year', 'month',
                'emergency_admissions', 'elective_total',
                'outpatient_first_total', 'outpatient_dna', 'gp_referrals']
    data = hes[[c for c in hes_cols if c in hes.columns]].copy()

    if len(gp):
        data = data.merge(gp, on=['ccg', 'year', 'month'], how='left')

    if len(iuc):
        data = data.merge(iuc, on=['ccg', 'year', 'month'], how='left')

    pop_annual = population[['ccg', 'year', 'population', 'pop_65plus', 'pct_65plus']].drop_duplicates()
    data = data.merge(pop_annual, on=['ccg', 'year'], how='left')

    # Fingertips uses ONS ICB codes; the rest of the pipeline uses NHS codes.
    # Bridge via Sub-ICB (NHS) to parent ICB (ONS) from the population file, then
    # ICB (NHS) to ICB (ONS) from the ambulance ICB lookup sheet.
    if fingertips is not None and len(fingertips):
        xl = pd.ExcelFile(POPULATION_FILE)
        mid = next((s for s in xl.sheet_names if 'Mid' in s), xl.sheet_names[0])
        raw_pop = pd.read_excel(POPULATION_FILE, sheet_name=mid, header=None, skiprows=4)
        ccg_to_ons = {}
        for _, r in raw_pop.iterrows():
            m = re.search(r'-\s*([0-9A-Z]{3})\s*$', str(r[1]))
            if m:
                ccg_to_ons[m.group(1)] = str(r[2]).strip()
        data['icb_ons'] = data['ccg'].map(ccg_to_ons)

        amb_lk = pd.read_excel(AMBSYS_FILE, sheet_name='ICB lookup', skiprows=2, header=None)
        nhs_to_ons = dict(zip(amb_lk[1].astype(str).str.strip(), amb_lk[0].astype(str).str.strip()))

        ft = fingertips.copy()
        ft['icb_ons'] = ft['icb_code'].map(nhs_to_ons)
        ft_cols = [c for c in ft.columns if c.startswith('fx_')]
        ft_merge = ft[['icb_ons', 'year'] + ft_cols].dropna(subset=['icb_ons'])
        ft_merge['year'] = ft_merge['year'].astype(float)

        data = data.merge(ft_merge, on=['icb_ons', 'year'], how='left')
        data = data.drop(columns=['icb_ons'], errors='ignore')

    # scale volume features to per 10,000 population
    data = data.dropna(subset=['emergency_admissions', 'population'])
    for col in ['emergency_admissions', 'gp_appt_available', '111_offered', '111_answered',
                'elective_total', 'outpatient_first_total', 'outpatient_dna', 'gp_referrals']:
        if col in data.columns:
            data[col + '_rate'] = data[col] / data['population']

    month_order = {m: i+1 for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}
    data['month_num'] = data['month'].map(month_order)
    return data.sort_values(['ccg', 'year', 'month_num']).reset_index(drop=True)


if __name__ == '__main__':
    hes = load_hes(HES_FILES)
    population = load_population(POPULATION_FILE)
    gp = load_gp_data(GP_FOLDER)
    iuc_raw, icb_map = load_iuc(IUC_FOLDER)
    iuc = stratify_iuc_to_subicb(iuc_raw, icb_map, population)
    fingertips = load_fingertips_subicb(FINGERTIPS_FILE)

    master = build_master(hes, gp, iuc, population, fingertips)
    master.to_csv(OUTPUT_FILE, index=False)
    print(f'Saved {OUTPUT_FILE}: {master.shape[0]} rows, {master.ccg.nunique()} Sub-ICBs')

    # check: confirm the previously-missing columns are now populated
    print('\nColumn coverage (non-null %):')
    for c in ['emergency_admissions_rate', 'gp_appt_available_rate',
              'elective_total_rate', 'outpatient_first_total_rate',
              'outpatient_dna_rate', 'gp_referrals_rate',
              'pct_65plus', 'population']:
        if c in master.columns:
            pct = master[c].notna().mean() * 100
            mean = master[c].mean()
            print(f'  {c}: {pct:.0f}% populated, mean = {mean:.3f}')
        else:
            print(f'  {c}: MISSING')
