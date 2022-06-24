from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.traceback import install
install()


console = Console(record=False)


def save_log(path='log.html'):
    console.save_html(path)


def new_progress():
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        refresh_per_second=2,
    )

    return progress


if __name__ == '__main__':
    from rich.table import Table
    from rich.align import Align
    from rich.progress_bar import ProgressBar
    from rich.live import Live
    import time

    bar = ProgressBar(width=100, total=1)
    table = Table('n', 'n**2', 'n**3')
    table_centered = Align.center(table)

    live = Live(
        table, console=console, screen=False, 
        refresh_per_second=4, vertical_overflow="visible"
    )

    total = 1000

    with live:
        for i in range(1, total+1):
            bar.update(bar.completed + 1, total)
            table.caption = bar
            table.add_row(f"{i}", f"{i**2}", f"{i**3}")
            time.sleep(0.1)