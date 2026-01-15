"""Position PnL Report - View virtual trading positions and statistics."""

import asyncio
import sys
import io
from datetime import datetime
from detector.storage import Storage
from detector.models import PositionStatus

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def format_price(price: float) -> str:
    """Format price with appropriate decimal places based on magnitude."""
    if price >= 100:
        return f"${price:,.2f}"
    elif price >= 10:
        return f"${price:,.3f}"
    elif price >= 1:
        return f"${price:,.4f}"
    elif price >= 0.01:
        return f"${price:,.5f}"
    elif price >= 0.0001:
        return f"${price:,.6f}"
    else:
        return f"${price:,.8f}"

async def print_positions():
    """Print open and closed positions with PnL stats."""
    storage = Storage("./data/market.db", wal_mode=False)
    await storage.init_db()

    # Get open positions
    open_positions = await storage.get_open_positions()
    closed_positions = await storage.query_positions(status=PositionStatus.CLOSED, limit=100)

    print("\n" + "="*80)
    print("VIRTUAL POSITION MANAGER - PnL REPORT")
    print("="*80)

    # Open positions
    print(f"\n📊 OPEN POSITIONS ({len(open_positions)})")
    print("-"*80)

    if open_positions:
        for pos in open_positions:
            duration_minutes = (int(datetime.now().timestamp() * 1000) - pos.open_ts) // (60 * 1000)
            print(f"\n{pos.symbol} {pos.direction.value}")
            print(f"  Position ID: {pos.position_id}")
            print(f"  Open Price:  {format_price(pos.open_price)}")
            print(f"  Entry Z-scores: ER={pos.entry_z_er:.2f}σ, VOL={pos.entry_z_vol:.2f}σ")
            print(f"  Entry Taker Share: {pos.entry_taker_share:.1%}")
            print(f"  MFE (Max Favorable):  {pos.max_favorable_excursion:+.2f}%")
            print(f"  MAE (Max Adverse):    {pos.max_adverse_excursion:+.2f}%")
            print(f"  Duration: {duration_minutes} minutes")
            print(f"  Event Status: {pos.metrics.get('event_status', 'N/A')}")
    else:
        print("  No open positions")

    # Closed positions summary
    print(f"\n\n💰 CLOSED POSITIONS ({len(closed_positions)})")
    print("-"*80)

    if closed_positions:
        # Calculate stats
        total_pnl = sum(p.pnl_percent for p in closed_positions if p.pnl_percent is not None)
        winning = [p for p in closed_positions if p.pnl_percent and p.pnl_percent > 0]
        losing = [p for p in closed_positions if p.pnl_percent and p.pnl_percent <= 0]

        win_rate = (len(winning) / len(closed_positions) * 100) if closed_positions else 0
        avg_win = (sum(p.pnl_percent for p in winning) / len(winning)) if winning else 0
        avg_loss = (sum(p.pnl_percent for p in losing) / len(losing)) if losing else 0
        avg_duration = (sum(p.duration_minutes for p in closed_positions if p.duration_minutes) / len(closed_positions)) if closed_positions else 0

        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  Total Positions:  {len(closed_positions)}")
        print(f"  Win Rate:         {win_rate:.1f}% ({len(winning)}W / {len(losing)}L)")
        print(f"  Total PnL:        {total_pnl:+.2f}%")
        print(f"  Avg Win:          {avg_win:+.2f}%")
        print(f"  Avg Loss:         {avg_loss:+.2f}%")
        print(f"  Avg Duration:     {avg_duration:.0f} minutes")

        # Exit reason breakdown
        print(f"\n🚪 EXIT REASONS:")
        exit_counts = {}
        for p in closed_positions:
            if p.exit_reason:
                reason = p.exit_reason.value
                exit_counts[reason] = exit_counts.get(reason, 0) + 1

        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(closed_positions)) * 100
            print(f"  {reason:25} {count:3} ({percentage:.1f}%)")

        # Recent positions (last 10)
        print(f"\n\n📋 RECENT CLOSED POSITIONS (Last 10):")
        print("-"*80)

        for pos in closed_positions[:10]:
            open_dt = datetime.fromtimestamp(pos.open_ts / 1000).strftime('%m-%d %H:%M')
            close_dt = datetime.fromtimestamp(pos.close_ts / 1000).strftime('%m-%d %H:%M')

            pnl_icon = "✅" if pos.pnl_percent > 0 else "❌"
            print(f"\n{pnl_icon} {pos.symbol} {pos.direction.value} | PnL: {pos.pnl_percent:+.2f}%")
            print(f"   Open:  {open_dt} @ {format_price(pos.open_price)}")
            print(f"   Close: {close_dt} @ {format_price(pos.close_price)}")
            print(f"   Duration: {pos.duration_minutes}m | Exit: {pos.exit_reason.value if pos.exit_reason else 'N/A'}")
            print(f"   MFE: {pos.max_favorable_excursion:+.2f}% | MAE: {pos.max_adverse_excursion:+.2f}%")

    else:
        print("  No closed positions yet")

    print("\n" + "="*80 + "\n")

    await storage.close()

if __name__ == "__main__":
    try:
        asyncio.run(print_positions())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
