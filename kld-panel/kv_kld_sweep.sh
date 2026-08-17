#!/bin/bash
# kv_kld_sweep.sh — KLD/PPL regression of custom KV-cache types vs logit bases.
#
# A run directory is an immutable campaign. On resume, the binaries, model, dataset,
# bases, types, and inference settings must match its recorded manifest exactly.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_args=()
for _a in "$@"; do
	case "$_a" in
		--shallow) export KV_SHALLOW=1 ;;
		*) _args+=("$_a") ;;
	esac
done
if [ ${#_args[@]} -gt 0 ]; then set -- "${_args[@]}"; else set --; fi

. "$HERE/kv_common.sh"

if [ -n "${TURBO_SCORE_LAST_K:-}" ] || [ -n "${TURBO_SCORE_LAST_ONLY:-}" ]; then
	echo "kv_kld_sweep.sh reports full-window percentiles and refuses inherited last-K scoring." >&2
	echo "Use hook-generated dumps plus the offline reducers for a last-K panel." >&2
	exit 2
fi
if [ -n "${TURBO_KLD_DUMP:-}" ]; then
	echo "Refusing inherited TURBO_KLD_DUMP: one fixed path would be overwritten by every cell." >&2
	exit 2
fi

RUN_DIR=${1:-${RUN_DIR:-./kld_$(date +%Y%m%d_%H%M%S)}}
TSV="$RUN_DIR/results.tsv"
MD="$RUN_DIR/results.md"
META="$RUN_DIR/meta.txt"
MANIFEST="$RUN_DIR/manifest.txt"
CURRENT="$RUN_DIR/current"
BPW_TSV="$RUN_DIR/bpw.tsv"
TYPES=${TYPES:-$(kv_special_types | tr '\n' ' ')}

if [ -z "$(printf '%s' "$TYPES" | tr -d '[:space:]')" ]; then
	echo "No custom KV-cache types found. Set TYPES explicitly or check $PPL_BIN --help." >&2
	exit 2
fi

mkdir -p "$RUN_DIR/logs"

write_manifest() {
	printf 'manifest_version=2\n'
	printf 'kind=KLD/PPL\n'
	printf 'ppl_bin=%s\n' "$(kv_file_identity "$PPL_BIN")"
	printf 'bench_bin=%s\n' "$(kv_file_identity "$BENCH_BIN")"
	printf 'build_sha=%s\n' "$(kv_build_sha)"
	printf 'model=%s\n' "$(kv_file_identity "$MODEL")"
	printf 'dataset=%s\n' "$(kv_file_identity "$DATASET")"
	printf 'base_dir=%s\n' "$(realpath -m "$BASE_DIR")"
	printf 'types=%s\n' "$TYPES"
	printf 'ktype=%s\n' "${KTYPE:-<matched K=V>}"
	printf 'mandatory_anchor=f16/f16\n'
	printf 'fa=%s\n' "$FA"
	printf 'ngl=%s\n' "$NGL"
	printf 'bpw_probe_ctx=%s\n' "$BPW_PROBE_CTX"
	printf 'allow_bpw_probe_failure=%s\n' "${ALLOW_BPW_PROBE_FAILURE:-0}"
	printf 'require_raw_anchor=%s\n' "${REQUIRE_RAW_ANCHOR:-1}"
	printf 'tiers_begin\n'
	while IFS=: read -r ctx base chunks; do
		[ -z "$ctx" ] && continue
		printf '%s:%s:%s:%s\n' "$ctx" "$base" "$chunks" "$(kv_file_identity "$BASE_DIR/$base")"
	done <<< "$KV_TIERS"
	printf 'tiers_end\n'
}

candidate=$(mktemp "$RUN_DIR/.manifest.XXXXXX")
write_manifest > "$candidate"
if [ -e "$MANIFEST" ]; then
	if ! cmp -s "$MANIFEST" "$candidate"; then
		echo "Refusing to resume: campaign inputs differ from $MANIFEST" >&2
		diff -u "$MANIFEST" "$candidate" >&2 || true
		rm -f "$candidate"
		exit 2
	fi
	rm -f "$candidate"
elif [ -s "$TSV" ] || [ -s "$META" ]; then
	echo "Refusing to adopt a pre-manifest run directory: $RUN_DIR" >&2
	echo "Use a new directory so results from different campaigns cannot mix." >&2
	rm -f "$candidate"
	exit 2
else
	mv "$candidate" "$MANIFEST"
fi

if [ ! -s "$META" ]; then
	{
		echo "started=$(date -u +%FT%TZ)"
		echo "gpu=$(kv_gpu)"
		cat "$MANIFEST"
	} > "$META"
fi

echo "Probing effective BPW (allocation/substitution guard) ..."
BPW_MAP=$(kv_build_bpw_tsv "$BPW_TSV" "${KTYPE:-}" $TYPES)
echo "  $BPW_MAP"
if awk -F'\t' 'NR>1 && ($10=="NOCAL" || $10=="PROBEFAIL"){bad=1} END{exit !bad}' "$BPW_TSV"; then
	if [ "${ALLOW_BPW_PROBE_FAILURE:-0}" != 1 ]; then
		echo "Effective-BPW validation failed; refusing to run without the substitution guard." >&2
		echo "Set ALLOW_BPW_PROBE_FAILURE=1 only when allocation accounting is intentionally unavailable." >&2
		exit 2
	fi
	echo "WARNING: effective-BPW validation was explicitly bypassed." >&2
fi

header='type	ctx	chunks	status	mean_kld	mean_kld_se	median_kld	p999_kld	max_kld	ppl_q	ln_ratio	rms_dp	same_top_p	secs'
if [ ! -s "$TSV" ]; then
	printf '%b\n' "$header" > "$TSV"
elif [ "$(head -n1 "$TSV")" != "$(printf '%b' "$header")" ]; then
	echo "Refusing to resume: unrecognized results.tsv schema in $RUN_DIR" >&2
	exit 2
fi

cell_done_ok() {
	awk -F'\t' -v t="$1" -v c="$2" 'NR>1 && $1==t && $2==c && $4=="OK"{found=1} END{exit !found}' "$TSV"
}

drop_cell_rows() {
	local tmp
	tmp=$(mktemp "$RUN_DIR/.results.XXXXXX")
	awk -F'\t' -v t="$1" -v c="$2" 'NR==1 || !($1==t && $2==c)' "$TSV" > "$tmp"
	mv "$tmp" "$TSV"
}

render_md() {
	{
		echo "# KV-cache KLD/PPL sweep"
		echo
		sed 's/^/    /' "$META"
		echo
		kv_render_bpw_md "$BPW_TSV"
		echo "| type | ctx | status | mean KLD | mean SE | median KLD | 99.9% KLD | max KLD | PPL(Q) | ln(Q/base) | RMS Δp% | same-top-p% | secs |"
		echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
		awk -F'\t' 'NR>1{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",$1,$2,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14}' "$TSV"
	} > "$MD"
}

run_cell() {
	local type=$1 ctx=$2 base=$3 chunks=$4 anchor=${5:-0}
	local ktype="${KTYPE:-$type}"
	local label="$type"; [ -n "${KTYPE:-}" ] && label="${KTYPE}:$type"
	if [ "$anchor" = 1 ]; then
		ktype=f16
		type=f16
		label=anchor_f16
	fi
	local log="$RUN_DIR/logs/${label//:/_}_ctx${ctx}.log"
	local basepath="$BASE_DIR/$base"
	local anchor_dump="$RUN_DIR/logs/anchor_f16_ctx${ctx}.kld"
	drop_cell_rows "$label" "$ctx"

	if [ ! -f "$basepath" ]; then
		printf '%s\t%s\t%s\tNOBASE\t-\t-\t-\t-\t-\t-\t-\t-\t-\t0\n' "$label" "$ctx" "$chunks" >> "$TSV"
		echo "  [skip] $label ctx=$ctx — base missing: $basepath"
		return
	fi

	echo "$label ctx=$ctx started=$(date -u +%FT%TZ) log=$log" > "$CURRENT"
	local t0 t1 rc status
	t0=$(date +%s)
	local -a run_env=(env "LD_LIBRARY_PATH=$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
	if [ "$anchor" = 1 ]; then
		rm -f "$anchor_dump" "$anchor_dump.meta"
		run_env+=("TURBO_KLD_DUMP=$anchor_dump")
	fi
	timeout "$CELL_TIMEOUT" "${run_env[@]}" "$PPL_BIN" \
		-m "$MODEL" -f "$DATASET" \
		--kl-divergence-base "$basepath" --kl-divergence \
		-ctk "$ktype" -ctv "$type" -fa "$FA" -c "$ctx" --chunks "$chunks" -ngl "$NGL" \
		> "$log" 2>&1
	rc=$?
	t1=$(date +%s)
	status=$(kv_classify "$rc" "$log")

	local mean mean_se median p999 maxk pplq lnr rms stp
	if [ "$status" = OK ]; then
		mean=$(kv_num "$log" 'Mean[[:space:]]+KLD:')
		mean_se=$(grep -m1 -E 'Mean[[:space:]]+KLD:' "$log" 2>/dev/null | grep -oE '[-]?[0-9]+\.[0-9]+' | sed -n '2p')
		median=$(kv_num "$log" 'Median[[:space:]]+KLD:')
		p999=$(kv_num "$log" '99\.9%[[:space:]]+KLD:')
		maxk=$(kv_num "$log" 'Maximum[[:space:]]+KLD:')
		pplq=$(kv_num "$log" 'Mean PPL\(Q\)')
		lnr=$(kv_num "$log" 'Mean ln\(PPL\(Q\)/PPL\(base\)\)')
		rms=$(kv_num "$log" 'RMS.*:')
		stp=$(kv_num "$log" 'Same top p:')
		if [ -z "$mean" ] || [ -z "$mean_se" ] || [ -z "$median" ] || [ -z "$p999" ] || \
		   [ -z "$maxk" ] || [ -z "$pplq" ] || [ -z "$lnr" ] || [ -z "$rms" ] || [ -z "$stp" ]; then
			status=PARSEFAIL
		fi
		if [ "$status" = OK ] && [ "$anchor" = 1 ] && ! awk \
			-v mean="$mean" -v se="$mean_se" -v median="$median" -v p999="$p999" \
			-v maxk="$maxk" -v lnr="$lnr" -v rms="$rms" -v stp="$stp" \
			'BEGIN{exit !(mean==0 && se==0 && median==0 && p999==0 && maxk==0 && lnr==0 && rms==0 && stp==100)}'; then
			status=ANCHORFAIL
		fi
		if [ "$status" = OK ] && [ "$anchor" = 1 ]; then
			if [ -f "$anchor_dump" ]; then
				if ! "$HERE/validate_kld_dump.py" --exact-zero "$anchor_dump" >> "$log" 2>&1; then
					status=ANCHORFAIL
				fi
			elif [ "${REQUIRE_RAW_ANCHOR:-1}" = 1 ]; then
				status=ANCHORUNVERIFIED
			fi
		fi
	fi
	if [ "$status" != OK ]; then
		mean=- mean_se=- median=- p999=- maxk=- pplq=- lnr=- rms=- stp=-
	fi

	printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
		"$label" "$ctx" "$chunks" "$status" "$mean" "$mean_se" "$median" "$p999" "$maxk" "$pplq" "$lnr" "$rms" "$stp" "$((t1 - t0))" >> "$TSV"
	render_md
	echo "  [$status] $label ctx=$ctx  meanKLD=$mean  PPL=$pplq  ($((t1 - t0))s)"
}

echo "KLD/PPL sweep -> $RUN_DIR"
echo "types: $TYPES"
while IFS=: read -r ctx base chunks; do
	[ -z "$ctx" ] && continue
	if cell_done_ok anchor_f16 "$ctx"; then
		echo "  [done] anchor_f16 ctx=$ctx (resume skip)"
		continue
	fi
	run_cell f16 "$ctx" "$base" "$chunks" 1
done <<< "$KV_TIERS"
for type in $TYPES; do
	label="$type"; [ -n "${KTYPE:-}" ] && label="${KTYPE}:$type"
	while IFS=: read -r ctx base chunks; do
		[ -z "$ctx" ] && continue
		if cell_done_ok "$label" "$ctx"; then
			echo "  [done] $label ctx=$ctx (resume skip)"
			continue
		fi
		run_cell "$type" "$ctx" "$base" "$chunks"
	done <<< "$KV_TIERS"
done

rm -f "$CURRENT"
printf 'finished=%s\n' "$(date -u +%FT%TZ)" > "$RUN_DIR/finished.txt"
render_md
echo "DONE. Results: $MD"
if awk -F'\t' 'NR>1 && $4!="OK"{bad=1} END{exit !bad}' "$TSV"; then
	echo "One or more cells did not complete successfully; see $TSV." >&2
	exit 2
fi
