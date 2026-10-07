# Tracciabilità delle Affermazioni (Claims Audit Map)

Questo documento fornisce la mappatura puntuale tra **ogni affermazione tecnica** dichiarata nel [README.md](../README.md) e il relativo test automatizzato o script di benchmark eseguibile (con percorso file e intervallo di righe esatto).

Nessuna proprietà architetturale o metrica di JANUS è presentata senza una corrispondente verifica automatizzata nella test suite (`pytest -v`).

---

## 1. Sintesi GBNF & Decodifica Vincolata (RFC 8259)

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Sintesi GBNF conforme alla RFC 8259:** escape completi (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\uXXXX`) ed esclusione dei caratteri di controllo ASCII `U+0000–U+001F`. | [`tests/test_gbnf_gen.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_gbnf_gen.py#L21-L32) | L21-L32 | `test_gbnf_string_rules_and_escapes` verifica le regole della stringa JSON. |
| **Distinzione numerica tra integer e float:** interi senza leading zeros (`007` vietato); `number` con frazione ed esponente opzionali. | [`tests/test_gbnf_gen.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_gbnf_gen.py#L33-L62) | L33-L62 | `test_gbnf_integer_vs_number_mapping` mappa `i8/i32/i64` su `integer` e `f16/f32` su `number`. |
| **Tipi complessi e assenza di fallback silenzioso:** tipi non supportati sollevano `GBNFGenerationError`; `ListType` e `CustomType` generano regole strutturate. | [`tests/test_gbnf_gen.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_gbnf_gen.py#L63-L101) | L63-L101 | `test_gbnf_unsupported_type_raises_clear_error`, `test_gbnf_custom_type_and_list_type_generation`. |
| **Ordinamento canonico parametri:** parametri obbligatori emessi per primi in ordine di dichiarazione, opzionali emessi con virgola iniziale. | [`tests/test_gbnf_gen.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_gbnf_gen.py#L102-L145) | L102-L145 | `test_gbnf_canonical_parameter_ordering_no_leading_comma`, `test_gbnf_all_optional_parameters_disjunctive_alternatives`. |
| **Grammatica agentica multi-tool DSL:** supporta `if/else`, accesso a campi record (`ident.field`) e spazio obbligatorio dopo `ret`. | [`tests/test_gbnf_gen.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_gbnf_gen.py#L146-L178) | L146-L178 | `test_gbnf_agent_grammar_supports_if_else_and_field_access`. |
| **CLI modalità di generazione (`call`, `output`, `agent`):** rimozione campo arbitrario `status: ok` salvo `--with-status`. | [`tests/test_gbnf_gen.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_gbnf_gen.py#L179-L251) | L179-L251 | `test_gbnf_output_grammar_clean_status`, `test_cli_gbnf_modes`. |
| **Validazione differenziale formale GBNF:** campionamento da grammatica supera `json.loads` e `jsonschema`; istanze invalide respinte da `GBNFRecognizer`. | [`tests/test_differential_gbnf.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_differential_gbnf.py#L20-L157) | L20-L157 | `test_sampling_differential_500_samples`, `test_hypothesis_valid_json_accepted_by_recognizer`, `test_negative_instances_rejected_by_recognizer`. |

---

## 2. Esportazione JSON Schema (Draft-07)

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Esportazione da Schema JANUS a JSON Schema Draft-07:** supporto modalità `call` e `output`, tipi primitivi, liste e `CustomType`. | [`janus/jsonschema_export.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/janus/jsonschema_export.py#L1-L120) | L1-L120 | Modulo principale con funzione `schema_to_json_schema`. |
| **CLI `janusc export-jsonschema`:** esportazione verso file o stdout formattato. | [`tests/test_lexer.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_lexer.py#L140-L181) | L140-L181 | Test end-to-end CLI con exit code 0 e diagnostica su file assente (code 2). |

---

## 3. Type Checking, Arity & Output Record Typing

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Controllo tipo argomenti tool:** widening `int -> float` ammesso, narrowing `float -> int` respinto (`ERR_TOOL_ARG_TYPE_MISMATCH`). | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L230-L284) | L230-L284 | `test_tool_arg_type_mismatch_primitive`, `test_tool_arg_widening_int_to_float_allowed`, `test_tool_arg_narrowing_float_to_int_rejected`. |
| **Controllo arità e argomenti duplicati:** solleva `ERR_TOOL_ARITY` e `ERR_DUPLICATE_TOOL_ARGUMENT`. | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L343-L397) | L343-L397 | `test_tool_duplicate_named_args`, `test_tool_duplicate_positional_and_named_args`, `test_tool_positional_arity_error`. |
| **Tipizzazione formale output tool:** ritorno `ToolOutputType` e propagazione su `res.field` con reiezione campi sconosciuti (`ERR_UNKNOWN_OUTPUT_FIELD`). | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L398-L471) | L398-L471 | `test_output_field_access_success_and_propagation`, `test_output_field_access_unknown_field_error`, `test_output_field_propagated_type_mismatch`. |

---

## 4. Effect System & Propagazione Transitiva

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Reticolo degli effetti (`pure <= stoc <= io`):** `pure` non può chiamare né direttamente né indirettamente `io` o `stoc` (`ERR_EFFECT_PURITY_VIOLATION`). | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L50-L153) | L50-L153 | `test_transitive_effect_direct_tool_call_violation`, `test_transitive_effect_indirect_violation`, `test_transitive_effect_deep_chain`. |
| **Chiusura transitiva su ricorsione mutua e chiamate annidate:** rilevamento violazioni di purezza attraverso cicli di chiamata. | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L154-L229) | L154-L229 | `test_transitive_effect_mutual_recursion`, `test_transitive_effect_nested_call_violation`. |
| **Escalation di effetto non permessa per `stoc`:** funzione `stoc` non può chiamare tool `io` o agenti (`ERR_EFFECT_ESCALATION`). | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L472-L548) | L472-L548 | `test_effect_lattice_stoc_calling_io_tool_escalation`, `test_effect_lattice_stoc_calling_io_func_escalation`, `test_effect_lattice_stoc_calling_agent_escalation`. |
| **Invocazione funzioni non definite:** solleva errore statico `ERR_UNDEFINED_FUNCTION`. | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L549-L561) | L549-L561 | `test_undefined_function_call_error`. |
| **CLI `--effects`:** tabella riassuntiva calcolata e output JSON strutturato. | [`tests/test_type_checker.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_type_checker.py#L610-L697) | L610-L697 | `test_cli_check_effects_plain_text`, `test_cli_check_effects_json_output`. |

---

## 5. Lexer Robusto & Fuzz Testing

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Classe `LexError` con coordinate:** intercetta caratteri illegali, numeri malformati (`1e`, `1.2.3`), stringhe non chiuse a newline o EOF. | [`tests/test_lexer.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_lexer.py#L25-L95) | L25-L95 | `test_illegal_characters_raise_lex_error`, `test_malformed_numbers_raise_lex_error`, `test_unclosed_strings_raise_lex_error`. |
| **Stringhe con apici singoli e operatori:** supporto per `'...'`, `%`, `^`, `!`. | [`tests/test_lexer.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_lexer.py#L96-L119) | L96-L119 | `test_single_quoted_strings`, `test_additional_operators`. |
| **Fuzz testing Hypothesis:** nessun crash inatteso (`assert` o `IndexError`) su testo casuale arbitrario; solo `LexError` o `ParseError`. | [`tests/test_lexer.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_lexer.py#L185-L197) | L185-L197 | `test_fuzz_lexer_and_parser`. |

---

## 6. Runtime Agentico, Validazione Bidirezionale & Sandbox

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **True dry-run:** esegue i tool `pure` registrati; mocka i tool `io`/`stoc` restituendo valori tipizzati conformi allo schema con `mocked=True`. | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L349-L393) | L349-L393 | `test_true_dry_run_pure_executed_and_io_mocked`. |
| **Validazione runtime bidirezionale:** convalida argomenti prima della chiamata e valori restituiti dopo la chiamata (`ToolValidationError`). | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L277-L348) | L277-L348 | `test_tool_validation_error_on_invalid_inputs`, `test_tool_validation_error_on_missing_required_input`, `test_tool_validation_error_on_invalid_outputs`. |
| **Verifica coerenza effetto a runtime:** disallineamento tra effetto registrato ed effetto nello schema solleva `ToolEffectMismatchError`. | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L253-L276) | L253-L276 | `test_schema_effect_mismatch_raises_error`. |
| **Fail-closed su tool non registrati:** solleva `ToolNotFoundError` nella sandbox e `JanusToolNotRegistered` nel codice generato eseguito standalone. | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L394-L441) | L394-L441 | `test_fail_closed_unregistered_tool_raises_not_found`, `test_fail_closed_unregistered_tool_standalone_codegen`. |
| **Ripristino garantito stato sandbox:** flag `dry_run` e `strict_effects` ripristinati in blocco `try/finally` anche in caso di eccezione. | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L442-L459) | L442-L459 | `test_agent_runtime_sandbox_state_restoration`. |

---

## 7. Strict Effects: Difesa Cooperativa & Concorrenza Thread-Safe

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Intercettazione I/O in tool pure:** `builtins.open`, `socket`, `subprocess`, `os.system` sollevano `StrictPurityViolationError`. | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L134-L179) | L134-L179 | `test_strict_effects_pure_tool_raises_on_file_io`, `test_strict_effects_pure_tool_raises_on_subprocess`, `test_strict_effects_pure_tool_raises_on_socket`. |
| **Isolamento thread-safe concorrente:** il guard è isolato sul thread corrente via `threading.local()`; thread concorrenti eseguono legittimo I/O senza blocco. | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L460-L504) | L460-L504 | `test_strict_effects_concurrency_thread_safety`. |
| **Limite noto e disclaimer (bypass ctypes):** test dichiarato esplicitamente come `xfail` che documenta come chiamate C di basso livello non siano intercettate dall'interprete Python in-process. | [`tests/test_agent_runtime.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_runtime.py#L505-L521) | L505-L521 | `test_strict_effects_ctypes_bypass_known_limitation` (`@pytest.mark.xfail`). |

---

## 8. Benchmark Empirici: Tool Calling ed Efficienza dei Token

| Affermazione nel README | File di Test / Benchmark / Risultato | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Valutazione Agentic Tool Calling (20 task):** decodifica vincolata GBNF garantisce 100% validità sintattica e rispetto dello schema per costruzione grammaticale. | [`tests/test_agent_eval.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_agent_eval.py#L63-L97) | L63-L97 | `test_agent_eval_benchmark_execution`, `test_agent_eval_multi_seed_and_conditions`. |
| **Risultati simulati dichiarati ed etichettati:** prefisso `simulated_` nei file e metadati `is_simulated: true`, `commit`, `janus_version`. | [`benchmarks/agent_eval.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/benchmarks/agent_eval.py#L880-L945) | L880-L945 | Generazione file JSON con metadati e assenza di tabelle hardcoded prefabbricate. |
| **Risultato negativo sui token BPE:** JANUS consuma +52.2% di token rispetto a Python equivalente sui tokenizer BPE standard `cl100k_base`. | [`benchmarks/results/tokens_benchmark.json`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/benchmarks/results/tokens_benchmark.json#L1-L180) | L1-L180 | Script `benchmarks/tokens.py`, validato in [`tests/test_no_hardcoded_benchmarks.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_no_hardcoded_benchmarks.py#L36-L48). |

---

## 9. Backend Tensoriale & Coerenza Numerica PyTorch

| Affermazione nel README | File di Test / Benchmark | Righe | Verifica Effettuata |
| :--- | :--- | :--- | :--- |
| **Equivalenza numerica rispetto a PyTorch:** 10 programmi di riferimento (Linear Regression, MLP, Attention, Conv2D, RMSNorm, Training Loop, Diffusion Denoise) convergono numericamente con scarto nullo. | [`tests/test_execution_reference.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_execution_reference.py#L35-L219) | L35-L219 | `test_linreg_convergence`, `test_mlp_matches_pytorch_reference`, `test_attention_matches_pytorch_reference`, `test_conv2d_matches_pytorch_reference`, `test_rmsnorm_matches_pytorch_reference`, `test_training_loop_execution`, `test_diffusion_denoise_execution`. |
| **Transpilazione priva di errori su tutti gli esempi:** tutti i 10 programmi di libreria ed esempi compilano in codice Python valido. | [`tests/test_no_hardcoded_benchmarks.py`](file:///Users/gianlucabernardo/Desktop/Gianluca/Progetti/janus/tests/test_no_hardcoded_benchmarks.py#L49-L71) | L49-L71 | `test_all_10_examples_compile_cleanly`. |
