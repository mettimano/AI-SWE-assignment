"""CLI entrypoint for Lumé.

Usage:
  python -m lume.cli "vorrei un profumo floreale per mia madre, budget 80€"
  python -m lume.cli --repl --user giulia
  python -m lume.cli --build-index
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from lume.agents.graph import run_turn
from lume.agents.intent import Intent
from lume.catalog.models import NormalizedProduct
from lume.schemas import Reply

app = typer.Typer(add_completion=False)
console = Console()


# ── Cross-turn state ──────────────────────────────────────────────────────────

@dataclass
class _Session:
    user_id: str | None = None
    current_intent: Intent | None = None
    last_shown_product_ids: list[str] = field(default_factory=list)
    last_shown_products: list[NormalizedProduct] = field(default_factory=list)
    topic_messages: list[str] = field(default_factory=list)
    clarify_count: int = 0
    last_action: str | None = None
    last_shown_mode: str | None = None

    def update_from(self, state: dict[str, Any]) -> None:
        for attr in (
            "current_intent",
            "last_shown_product_ids",
            "last_shown_products",
            "topic_messages",
            "clarify_count",
            "last_action",
            "last_shown_mode",
        ):
            if attr in state:  # preserve None resets (e.g. last_shown_mode after new_topic)
                setattr(self, attr, state[attr])

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "current_intent": self.current_intent,
            "last_shown_product_ids": self.last_shown_product_ids,
            "last_shown_products": self.last_shown_products,
            "topic_messages": self.topic_messages,
            "clarify_count": self.clarify_count,
            "last_action": self.last_action,
        }


# ── Display helpers ───────────────────────────────────────────────────────────

_MODE_STYLE: dict[str, str] = {
    "answer": "bold green",
    "clarify_question": "bold yellow",
    "clarify_probe": "bold cyan",
    "escalate": "bold red",
    "no_match": "bold magenta",
    "specification": "bold blue",
}


def _display_reply(reply: Reply, debug: bool = False) -> None:
    mode_style = _MODE_STYLE.get(reply.mode, "bold white")
    mode_label = Text(f" {reply.mode} ", style=f"on {mode_style.split()[1]}")

    # Main reply panel
    console.print(
        Panel(
            reply.reply_text,
            title=mode_label,
            border_style=mode_style.split()[1],
            padding=(0, 1),
        )
    )

    # Recommendations table
    if reply.recommendations:
        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
            title="Prodotti raccomandati",
            title_style="dim",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Prodotto", min_width=30)
        table.add_column("Prezzo", justify="right")
        table.add_column("Disp.", justify="center", width=6)
        table.add_column("Perché", min_width=30)
        for i, rec in enumerate(reply.recommendations, 1):
            avail = "[green]✓[/green]" if rec.available else "[red]✗[/red]"
            table.add_row(str(i), rec.title[:45], rec.price_display, avail, rec.why[:50])
        console.print(table)

    # Probe products (clarify_probe)
    if reply.probes:
        probe_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1),
                            title="Prodotti probe", title_style="dim")
        probe_table.add_column("#", style="dim", width=3)
        probe_table.add_column("Prodotto", min_width=30)
        probe_table.add_column("Stile", min_width=15)
        for i, p in enumerate(reply.probes, 1):
            probe_table.add_row(str(i), p.title[:45], p.axis_value)
        console.print(probe_table)

    # Clarify questions
    if reply.questions:
        for q in reply.questions:
            console.print(f"  [yellow]❓ {q.text}[/yellow]")

    # Guard warnings
    if reply.debug.get("guard_violations"):
        console.print(
            f"  [dim red]⚠ guard violations: {reply.debug['guard_violations']}[/dim red]"
        )
    if reply.debug.get("guard_regenerated"):
        console.print("  [dim yellow]↻ guard regenerated reply[/dim yellow]")
    if reply.debug.get("guard_fallback"):
        console.print("  [dim red]⚠ guard fallback applied[/dim red]")

    # Debug info
    if debug:
        latency = reply.debug.get("latency_ms", "?")
        console.print(f"  [dim]mode={reply.mode}  latency={latency}ms[/dim]")
        if reply.intent:
            console.print(f"  [dim]intent.confidence={reply.intent.confidence:.2f}  "
                          f"cats={reply.intent.categories}  "
                          f"family={reply.intent.fragrance_family}  "
                          f"budget={reply.intent.budget_max}[/dim]")


# ── Commands ──────────────────────────────────────────────────────────────────

def _run_one(message: str, session: _Session, debug: bool) -> None:
    with console.status("[dim]Elaboro...[/dim]", spinner="dots"):
        state = run_turn(message, **session.as_kwargs())
    session.update_from(state)
    reply: Reply | None = state.get("reply")
    if reply is None:
        console.print("[red]Nessuna risposta generata (errore interno).[/red]")
        return
    _display_reply(reply, debug=debug)


@app.command()
def main(
    query: str = typer.Argument(None, help="Query singola (one-shot)"),
    repl: bool = typer.Option(False, "--repl", "-r", help="Modalità REPL multi-turn"),
    user: str = typer.Option(None, "--user", "-u", help="User ID per memoria persistente"),
    build_index: bool = typer.Option(False, "--build-index", help="Costruisce indici Chroma e BM25"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Mostra info debug (intent, latency)"),
) -> None:
    """Lumé — assistente WhatsApp per beauty e profumeria."""

    if build_index:
        _cmd_build_index()
        return

    session = _Session(user_id=user)

    if repl or query is None:
        _cmd_repl(session, debug=debug)
    else:
        _run_one(query, session, debug=debug)


def _cmd_build_index() -> None:
    from lume.catalog.loader import load_products
    from lume.catalog.normalize import normalize_all
    from lume.config import CACHE_DIR, CATALOG_PATH
    from lume.retrieval.bm25 import BM25Index
    from lume.retrieval.vectors import build_index

    console.print("[bold]Costruendo indici...[/bold]")

    with console.status("Caricamento catalogo..."):
        products = normalize_all(load_products(CATALOG_PATH))
    console.print(f"  ✓ {len(products)} prodotti caricati")

    with console.status("Costruendo indice vettoriale (Chroma)..."):
        build_index(products, reset=False)
    console.print("  ✓ Indice vettoriale pronto")

    with console.status("Costruendo indice BM25..."):
        bm25 = BM25Index(products)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        bm25.save(CACHE_DIR / "bm25_index.pkl")
    console.print("  ✓ Indice BM25 salvato")

    console.print("[bold green]Indici pronti.[/bold green]")


def _cmd_repl(session: _Session, debug: bool) -> None:
    user_label = f"[cyan]{session.user_id}[/cyan]" if session.user_id else "[dim]anonimo[/dim]"
    console.print(
        Panel(
            f"Lumé WhatsApp Assistant  •  utente: {user_label}\n"
            "[dim]Scrivi il tuo messaggio. Comandi: /quit  /reset  /debug[/dim]",
            border_style="blue",
            padding=(0, 1),
        )
    )

    while True:
        try:
            raw = console.input("[bold blue]Tu:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Arrivederci![/dim]")
            break

        if not raw:
            continue

        if raw.lower() in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]Arrivederci![/dim]")
            break

        if raw.lower() == "/reset":
            session.__init__(user_id=session.user_id)  # type: ignore[misc]
            console.print("[dim]Sessione resettata.[/dim]")
            continue

        if raw.lower() == "/debug":
            debug = not debug
            console.print(f"[dim]Debug {'on' if debug else 'off'}.[/dim]")
            continue

        _run_one(raw, session, debug=debug)
        console.print()


# ── Module entry point ────────────────────────────────────────────────────────

def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
