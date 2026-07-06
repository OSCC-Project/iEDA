# compare_spef

`compare_spef` compares two SPEF files and writes reports for total capacitance,
ground capacitance, coupling capacitance, point-to-point resistance, mismatched
nets, and mismatched coupling capacitances.

This command currently supports SPEF input only.

## Command

Basic form:

```tcl
compare_spef test.spef reference.spef [options]
```

Optional arguments:

```text
-output_dir dir
-c
-r
-tcap value
-ccap abs_value rel_value
-res value
-cores value
-match name
-net net_name
-from_pin pin_name
-to_pin pin_name
-net_config file
```

`test.spef` is the SPEF being checked. `reference.spef` is the golden SPEF.

If neither `-c` nor `-r` is specified, the command compares both capacitance and
resistance.

## Common Examples

Compare capacitance and resistance with default thresholds:

```tcl
compare_spef new.spef golden.spef -output_dir cmp_rpt
```

Compare only capacitance:

```tcl
compare_spef new.spef golden.spef -c -output_dir cmp_rpt
```

Compare only point-to-point resistance:

```tcl
compare_spef new.spef golden.spef -r -res 50 -output_dir cmp_rpt
```

Compare one net:

```tcl
compare_spef new.spef golden.spef -net clk -output_dir cmp_rpt
```

Compare paths from one pin:

```tcl
compare_spef new.spef golden.spef -r -from_pin U1/Y -output_dir cmp_rpt
```

Compare one pin-to-pin path:

```tcl
compare_spef new.spef golden.spef -r -from_pin U1/Y -to_pin U2/A -output_dir cmp_rpt
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `-output_dir dir` | `.` | Directory for generated reports. It is created if needed. |
| `-c` | off | Compare capacitance only, unless combined with `-r`. |
| `-r` | off | Compare point-to-point resistance only, unless combined with `-c`. |
| `-tcap value` | `3.0` | Include total-cap rows whose reference total cap is at least this value. |
| `-ccap abs rel` | `0.3 0.1` | Include ground-cap rows by the absolute threshold, and coupling-cap rows whose reference coupling cap and relative coupling ratio both meet the thresholds. |
| `-res value` | `50.0` | Include point-to-point resistance rows whose reference resistance is at least this value. |
| `-cores value` | `1` | Number of threads used by the SPEF comparison stage. SPEF parsing and report writing remain serial. |
| `-match name` | `name` | Pin matching mode. Only `name` is supported. |
| `-net net_name` | none | Compare one net. Cannot be used with direct `-from_pin` or `-to_pin`. |
| `-from_pin pin` | none | Compare paths that start from this pin. |
| `-to_pin pin` | none | Compare paths that end at this pin. |
| `-net_config file` | none | Read multiple net/path filters from a file. |

The parsed but currently unsupported or unused options are:

| Option | Current behavior |
| --- | --- |
| `-corner` | Rejected. GPD/corner comparison is not supported. |
| `-match xy` | Rejected. Only name-based matching is supported. |
| `-d`, `-delay`, `-delay_pin_load` | Elmore delay comparison is not implemented. |
| `-timeout` | Parsed for interface compatibility, but not used by the current SPEF implementation. |

## Filtering Rules

Capacitance comparison:

- `tcap.rpt` includes matched reference nets whose reference total capacitance is
  at least `-tcap`.
- `gcap.rpt` includes matched net ground caps when
  `abs(reference_gcap) >= -ccap abs` or `abs(test_gcap) >= -ccap abs`.
- `ccap.rpt` includes matched reference coupling pairs only when both conditions
  hold:
  - `abs(reference_ccap) >= -ccap abs`
  - `abs(reference_ccap) / reference_victim_total_cap >= -ccap rel`
  The `%diff` value in `ccap.rpt` is computed as
  `(test_ccap - reference_ccap) / reference_ccap`.

Resistance comparison:

- `p2p.rpt` includes pin pairs whose reference point-to-point resistance is at
  least `-res`.
- Without path filters, the command chooses a bounded default pin-pair set for
  each selected net.
- With `-from_pin`, `-to_pin`, or `FROM_TO_PINS`, only matching configured paths
  are considered.

Net selection:

- With no net/path filter, all matched nets are considered.
- `-net` selects one net.
- `-from_pin` or `-to_pin` selects nets containing the specified pin.
- `-net_config` can provide multiple net and path filters.

## Net Config File

`-net_config` accepts one statement per line:

```text
// comment
** comment
NET: net_name
FROM_PIN: start_pin
TO_PIN: end_pin
FROM_TO_PINS: start_pin end_pin
```

Example:

```text
** compare selected clock paths
NET: clk
FROM_PIN: U_CLKBUF/Y
FROM_TO_PINS: U1/Q U2/D
```

Blank lines and lines starting with `//` or `**` are ignored.

## Output Files

The command writes these files under `-output_dir`:

| File | Content |
| --- | --- |
| `summary.rpt` | Overview, thresholds, row counts, and error distributions. |
| `tcap.rpt` | Total capacitance differences. |
| `gcap.rpt` | Ground capacitance differences by net. |
| `ccap.rpt` | Coupling capacitance differences. |
| `p2p.rpt` | Point-to-point resistance differences. |
| `nets.mismatched` | Nets found only in reference or only in test. |
| `coupling_caps.mismatched` | Coupling-cap pairs found only in reference or only in test. |

## Current Limitations

- Only SPEF files are supported.
- Only name-based matching is supported.
- GPD, SMC corners, XY matching, and Elmore delay are not implemented.
- Threshold values are interpreted in the numeric units used by the current
  implementation and reports.
