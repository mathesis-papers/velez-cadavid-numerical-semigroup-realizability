import argparse
from collections import Counter
import hashlib
import heapq
import json
from math import gcd
from pathlib import Path


def apery(generators, modulus):
    distances = [None] * modulus
    distances[0] = 0
    queue = [(0, 0)]
    while queue:
        distance, residue = heapq.heappop(queue)
        if distance != distances[residue]:
            continue
        for generator in generators:
            candidate = distance + generator
            target = candidate % modulus
            if distances[target] is None or candidate < distances[target]:
                distances[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return distances


def semigroups(bound):
    for mask in range(1 << bound):
        gaps = {index + 1 for index in range(bound) if mask & (1 << index)}
        valid = True
        for gap in gaps:
            if any(left not in gaps and gap - left not in gaps
                   for left in range(1, gap)):
                valid = False
                break
        if valid:
            yield gaps


def generators_from_gaps(gaps):
    conductor = max(gaps, default=-1) + 1
    multiplicity = next(value for value in range(1, conductor + 2) if value not in gaps)
    largest = max(multiplicity, conductor + multiplicity - 1)
    generators = {value for value in range(1, largest + 1)
                  if value not in gaps and not any(
                      left not in gaps and value - left not in gaps
                      for left in range(1, value))}
    return conductor, multiplicity, generators


def check_realization(gaps, generators, maximum, tail):
    jumps = {maximum} | {maximum - generator for generator in tail}
    assert min(jumps) > 0 and max(jumps) == maximum
    divisor = 0
    for jump in jumps:
        divisor = gcd(divisor, jump)
    assert divisor == 1
    defects = {maximum} | {maximum - jump for jump in jumps} | {maximum + jump for jump in jumps}
    assert all(defect not in gaps for defect in defects)
    assert generators <= defects
    target = []
    for residue in range(maximum):
        value = residue
        while value in gaps:
            value += maximum
        target.append(value)
    observed = apery(defects - {0}, maximum)
    assert observed == target
    return {'gaps': sorted(gaps), 'M': maximum, 'S': sorted(jumps), 'apery': observed}


def target_apery(gaps, modulus):
    representatives = []
    for residue in range(modulus):
        value = residue
        while value in gaps:
            value += modulus
        representatives.append(value)
    return representatives


def defect_generators(jumps, maximum):
    return {maximum} | {maximum - jump for jump in jumps} | {maximum + jump for jump in jumps}


def criterion(gaps, generators, maximum):
    return maximum not in gaps and all(
        generator < 2 * maximum and 2 * maximum - generator not in gaps
        for generator in generators)


def minimum_jumps(generators, maximum):
    return {maximum} | {abs(maximum - generator) for generator in generators if generator != maximum}


def check_reconstruction(gaps, maximum):
    counts = [sum(gap % maximum == residue for gap in gaps) for residue in range(maximum)]
    assert counts[0] == 0
    if maximum == 1:
        assert not gaps
        return
    degree = 10000
    genus = len(gaps)
    global_now = maximum * degree + degree // 2 + 1 - maximum * (maximum + 1) - genus
    global_next = maximum * (degree + 2) + (degree + 2) // 2 + 1 - maximum * (maximum + 1) - genus
    recovered_maximum = (global_next - global_now - 1) // 2
    assert recovered_maximum == maximum
    recovered_genus = maximum * degree + degree // 2 + 1 - maximum * (maximum + 1) - global_now
    recovered = [0] * maximum
    for residue in range(1, maximum - 1):
        dimension = degree + degree // 2 - 1 - counts[residue]
        recovered[residue] = degree + degree // 2 - 1 - dimension
    recovered[-1] = recovered_genus - sum(recovered[:-1])
    assert recovered == counts
    assert [residue + maximum * count for residue, count in enumerate(recovered)] == target_apery(gaps, maximum)


def enumerate_jump_sets(bound):
    images = {}
    count = 0
    for maximum in range(1, bound + 1):
        images[maximum] = set()
        for mask in range(1 << (maximum - 1)):
            jumps = {maximum} | {index + 1 for index in range(maximum - 1) if mask & (1 << index)}
            if gcd(*jumps) != 1:
                continue
            defects = defect_generators(jumps, maximum)
            representatives = apery(defects - {0}, maximum)
            assert all(value is not None for value in representatives)
            gaps = frozenset(value for residue, least in enumerate(representatives)
                             for value in range(residue, least, maximum))
            _, _, generators = generators_from_gaps(gaps)
            assert criterion(gaps, generators, maximum)
            forced = minimum_jumps(generators, maximum)
            assert forced <= jumps
            assert apery(defect_generators(forced, maximum) - {0}, maximum) == representatives
            images[maximum].add(gaps)
            count += 1
    return images, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-frobenius', type=int, default=16)
    parser.add_argument('--max-bruteforce-maximum', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('verification.json'))
    args = parser.parse_args()
    if not 0 <= args.max_frobenius <= 20:
        parser.error('Choose a largest-gap bound from 0 through 20.')
    if not 1 <= args.max_bruteforce_maximum <= 16:
        parser.error('Choose a jump-set enumeration maximum from 1 through 16.')
    digest = hashlib.sha256()
    cases = 0
    reflected = 0
    consecutive = 0
    histogram = Counter()
    images, primitive_jump_sets = enumerate_jump_sets(args.max_bruteforce_maximum)
    criterion_tests = 0
    minimal_realizations = 0
    brute_comparisons = 0
    spectral_reconstructions = 0
    largest_tested_maximum = 0
    for gaps in semigroups(args.max_frobenius):
        conductor, multiplicity, generators = generators_from_gaps(gaps)
        assert generators and gcd(*generators) == 1
        assert all(value not in gaps for value in generators)
        threshold = max(conductor, 1 + max(generators))
        assert threshold <= max(2, conductor + multiplicity)
        for maximum in (threshold, threshold + 1, threshold + 2, threshold + 7):
            record = check_realization(gaps, generators, maximum, generators)
            assert len(record['S']) == len(generators) + 1
            digest.update(json.dumps(record, sort_keys=True, separators=(',', ':')).encode())
            reflected += 1
        improved = max(conductor, max(generators))
        assert improved <= max(1, conductor + multiplicity - 1)
        for maximum in (improved, improved + 1, improved + 2):
            record = check_realization(gaps, generators, maximum, {generator for generator in generators if generator < maximum})
            assert len(record['S']) == len(generators) + 1 - int(maximum in generators)
            reflected += 1
        for maximum in range(1, max(args.max_bruteforce_maximum, improved + 2) + 1):
            expected = criterion(gaps, generators, maximum)
            allowed = {jump for jump in range(1, maximum + 1)
                       if maximum - jump not in gaps and maximum + jump not in gaps}
            possible = False
            if maximum not in gaps and maximum in allowed:
                observed = apery(defect_generators(allowed, maximum) - {0}, maximum)
                possible = observed == target_apery(gaps, maximum)
            assert possible == expected
            criterion_tests += 1
            largest_tested_maximum = max(largest_tested_maximum, maximum)
            if maximum <= args.max_bruteforce_maximum:
                assert expected == (frozenset(gaps) in images[maximum])
                brute_comparisons += 1
            if expected:
                jumps = minimum_jumps(generators, maximum)
                assert jumps <= allowed and min(jumps) > 0 and max(jumps) == maximum
                assert gcd(*jumps) == 1
                assert apery(defect_generators(jumps, maximum) - {0}, maximum) == target_apery(gaps, maximum)
                for removed in jumps - {maximum}:
                    smaller = jumps - {removed}
                    assert apery(defect_generators(smaller, maximum) - {0}, maximum) != target_apery(gaps, maximum)
                check_reconstruction(gaps, maximum)
                spectral_reconstructions += 1
                minimal_realizations += 1
            digest.update(json.dumps({'gaps': sorted(gaps), 'M': maximum, 'admissible': expected}, sort_keys=True, separators=(',', ':')).encode())
        tail = generators | {conductor, conductor + 1}
        maximum = max(tail) + 1
        record = check_realization(gaps, generators, maximum, tail)
        digest.update(json.dumps(record, sort_keys=True, separators=(',', ':')).encode())
        consecutive += 1
        cases += 1
        histogram[str(max(gaps, default=-1))] += 1
    boundary = check_realization(set(), {1}, 1, set())
    digest.update(json.dumps(boundary, sort_keys=True, separators=(',', ':')).encode())
    ordinary_tests = 0
    for multiplicity in range(1, 25):
        gaps = set(range(1, multiplicity))
        _, _, generators = generators_from_gaps(gaps)
        threshold = (3 * multiplicity) // 2
        for maximum in range(1, 3 * multiplicity + 1):
            assert criterion(gaps, generators, maximum) == (maximum >= threshold)
            ordinary_tests += 1
    for maximum in (5, 7, 11, 13, 17, 19, 23, 29, 31):
        for gaps in ({1, 2}, {1, 3}):
            _, _, generators = generators_from_gaps(gaps)
            assert criterion(gaps, generators, maximum)
            jumps = minimum_jumps(generators, maximum)
            assert apery(defect_generators(jumps, maximum) - {0}, maximum) == target_apery(gaps, maximum)
    result = {
        'schema_version': 1,
        'status': 'all checks passed',
        'scope': 'All numerical semigroups with largest gap at most the stated bound, including N.',
        'max_frobenius': args.max_frobenius,
        'gap_masks_examined': 1 << args.max_frobenius,
        'numerical_semigroups': cases,
        'reflected_generator_realizations': reflected,
        'consecutive_tail_realizations': consecutive,
        'fixed_maximum_criterion_tests': criterion_tests,
        'largest_tested_maximum': largest_tested_maximum,
        'minimal_jump_set_realizations': minimal_realizations,
        'bruteforce_maximum': args.max_bruteforce_maximum,
        'primitive_jump_sets_enumerated': primitive_jump_sets,
        'criterion_vs_exhaustive_jump_sets': brute_comparisons,
        'residue_count_reconstructions': spectral_reconstructions,
        'spectral_check_scope': 'Exact inversion of the cited eventual dimension formulas; not evaluation of graph spectral ranks or a claim that degree 10000 is a stable threshold.',
        'ordinary_multiplicities': [1, 24],
        'ordinary_threshold_tests': ordinary_tests,
        'prime_obstruction_maxima': [5, 7, 11, 13, 17, 19, 23, 29, 31],
        'additional_N_maximum_one_case': 1,
        'semigroup_equality': 'Complete Apery representative lists modulo M agree.',
        'arithmetic': 'Exact unbounded integers; no floating-point operations.',
        'cases_by_largest_gap': dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        'ordered_witness_sha256': digest.hexdigest(),
        'coverage': 'Finite instances of the constructions, prescribed-maximum criterion, minimal jump sets, ordinary thresholds, and residue-count reconstruction; universal statements have written proofs in note.tex.',
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
