"""Optional operator explanation through NVIDIA Cosmos Reason (build.nvidia.com), with the deterministic template as the fallback.

Status: the operator message is a deterministic template (explain.py); this optional Cosmos Reason client exists but
is not called by the pipeline and has not been run.

The decision never depends on this module: the carryover flag is computed by decide.py from measurements. This only
turns the evidence of one vial into an operator-facing message. It calls the OpenAI-compatible chat completions API
that build.nvidia.com exposes for the hosted Cosmos 3 reasoner. Nothing is sent unless NVIDIA_API_KEY is set, and the
template is returned (with reason) whenever the call is unavailable or fails. Untested against the hosted endpoint on
this host: no key was available during development, so treat the first live call as a smoke test.

Environment:
  NVIDIA_API_KEY   required for a live call (free keys: https://build.nvidia.com/settings/api-keys)
  COSMOS_BASE_URL  default https://integrate.api.nvidia.com/v1
  COSMOS_MODEL     default: the first model id containing "cosmos" from GET /models, else "nvidia/cosmos3-nano-reasoner"
"""
import base64, json, os, urllib.request, urllib.error

DEFAULT_BASE = 'https://integrate.api.nvidia.com/v1'
DEFAULT_MODEL = 'nvidia/cosmos3-nano-reasoner'


def _request(url, key, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def discover_model(key, base):
    """Pick the served model id at runtime, as the Cosmos cookbook advises, instead of hardcoding it."""
    forced = os.environ.get('COSMOS_MODEL')
    if forced:
        return forced
    try:
        ids = [m.get('id', '') for m in _request(base + '/models', key).get('data', [])]
        cosmos = [i for i in ids if 'cosmos' in i.lower() and 'reason' in i.lower()] or [i for i in ids if 'cosmos' in i.lower()]
        return cosmos[0] if cosmos else DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def evidence_prompt(record):
    """The quantitative evidence, as text, for the model to explain; it must not add facts of its own."""
    r = record
    return (
        "You are writing a short operator note for a fluid-mixing line. Use ONLY the measurements below; do not invent "
        "numbers, causes or actions beyond flushing the shared dispensing path. Three sentences at most: what was measured, "
        "what it indicates, what to do.\n"
        f"Vial {r['vial']}, programmed recipe {r['recipe_id']}, previous recipe {r.get('prev_recipe_id')}, position in run {r.get('position_in_run')}.\n"
        f"Measured liquid colour (CIE Lab): {r.get('measured_lab')}. Settled colour of recipe {r['recipe_id']} from its later vials: {r.get('reference_lab')}.\n"
        f"Colour difference CIE76: {r.get('dE_vs_reference')} against the frozen threshold {r.get('tau')}. Decision by rule: {r['decision']}.\n"
        f"The image is the fill-station camera at the measurement moment, with the measured region outlined."
    )


def explain_with_cosmos(record, image_path=None, template_text=''):
    """Return (text, source). source is 'cosmos' or 'template:<reason>'."""
    key = os.environ.get('NVIDIA_API_KEY')
    if not key:
        return template_text, 'template:no NVIDIA_API_KEY'
    base = os.environ.get('COSMOS_BASE_URL', DEFAULT_BASE).rstrip('/')
    content = []
    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode('ascii')}})
    content.append({'type': 'text', 'text': evidence_prompt(record)})
    payload = {'model': discover_model(key, base), 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 1024, 'temperature': 0.2}
    try:
        out = _request(base + '/chat/completions', key, payload, timeout=120)
        text = out['choices'][0]['message']['content'].strip()
        # strip a reasoning preamble if the model returns one in <think> tags
        if '</think>' in text:
            text = text.split('</think>')[-1].strip()
        return (text or template_text), ('cosmos' if text else 'template:empty response')
    except urllib.error.HTTPError as e:
        return template_text, f'template:http {e.code}'
    except Exception as e:  # network, timeout, schema
        return template_text, f'template:{type(e).__name__}'
