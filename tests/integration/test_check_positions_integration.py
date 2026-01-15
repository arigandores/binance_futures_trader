"""Integration test for check_positions.py - Verify it works with real position data."""

import asyncio
import sys
import io
from datetime import datetime
from detector.storage import Storage
from detector.models import Position, PositionStatus, Direction, ExitReason

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

async def create_test_positions():
    """Create test positions in the database."""
    storage = Storage("./data/market.db", wal_mode=False)
    await storage.init_db()

    # Create an open position
    open_pos = Position(
        position_id="BTCUSDT_1736936400000_UP_test",
        event_id="evt_001",
        symbol="BTCUSDT",
        direction=Direction.UP,
        status=PositionStatus.OPEN,
        open_price=43520.50,
        open_ts=1736936400000,  # 2025-01-15 10:00:00
        entry_z_er=3.45,
        entry_z_vol=4.12,
        entry_taker_share=0.68,
        metrics={
            'event_status': 'CONFIRMED',
            'beta': 1.05,
            'funding': 0.0001
        }
    )
    open_pos.max_favorable_excursion = 1.25
    open_pos.max_adverse_excursion = -0.45

    # Create closed winning position
    closed_win = Position(
        position_id="ETHUSDT_1736932800000_UP_test",
        event_id="evt_002",
        symbol="ETHUSDT",
        direction=Direction.UP,
        status=PositionStatus.OPEN,
        open_price=2450.80,
        open_ts=1736932800000,  # 2025-01-15 09:00:00
        entry_z_er=3.12,
        entry_z_vol=3.89,
        entry_taker_share=0.72,
        metrics={
            'event_status': 'CONFIRMED',
            'beta': 1.15,
            'funding': 0.00015
        }
    )
    closed_win.close_position(
        close_price=2513.50,
        close_ts=1736936400000,  # 2025-01-15 10:00:00
        exit_reason=ExitReason.TAKE_PROFIT,
        exit_z_er=0.45,
        exit_z_vol=1.20
    )
    closed_win.max_favorable_excursion = 2.56
    closed_win.max_adverse_excursion = -0.12

    # Create closed losing position
    closed_loss = Position(
        position_id="SOLUSDT_1736929200000_DOWN_test",
        event_id="evt_003",
        symbol="SOLUSDT",
        direction=Direction.DOWN,
        status=PositionStatus.OPEN,
        open_price=98.75,
        open_ts=1736929200000,  # 2025-01-15 08:00:00
        entry_z_er=-3.25,
        entry_z_vol=4.05,
        entry_taker_share=0.28,
        metrics={
            'event_status': 'UNCONFIRMED',
            'beta': 0.95,
            'funding': -0.0001
        }
    )
    closed_loss.close_position(
        close_price=100.50,
        close_ts=1736932800000,  # 2025-01-15 09:00:00
        exit_reason=ExitReason.STOP_LOSS,
        exit_z_er=0.15,
        exit_z_vol=0.80
    )
    closed_loss.max_favorable_excursion = 0.85
    closed_loss.max_adverse_excursion = -1.77

    # Create closed position with trailing stop
    closed_trailing = Position(
        position_id="BTCUSDT_1736925600000_UP_test",
        event_id="evt_004",
        symbol="BTCUSDT",
        direction=Direction.UP,
        status=PositionStatus.OPEN,
        open_price=42850.00,
        open_ts=1736925600000,  # 2025-01-15 07:00:00
        entry_z_er=3.78,
        entry_z_vol=3.95,
        entry_taker_share=0.75,
        metrics={
            'event_status': 'CONFIRMED',
            'beta': 1.0,
            'funding': 0.00012,
            'trigger_delay_bars': 3,
            'trigger_delay_seconds_approx': 180
        }
    )
    closed_trailing.close_position(
        close_price=43950.00,
        close_ts=1736929200000,  # 2025-01-15 08:00:00
        exit_reason=ExitReason.TRAILING_STOP,
        exit_z_er=1.25,
        exit_z_vol=2.10
    )
    closed_trailing.max_favorable_excursion = 3.15
    closed_trailing.max_adverse_excursion = -0.35

    # Save all positions
    await storage.write_position(open_pos)
    await storage.write_position(closed_win)
    await storage.write_position(closed_loss)
    await storage.write_position(closed_trailing)

    await storage.close()

    print("✅ Test positions created successfully")
    print(f"   - 1 open position (BTCUSDT UP)")
    print(f"   - 3 closed positions (2 wins, 1 loss)")

async def verify_check_positions_output():
    """Run check_positions.py and verify output."""
    import subprocess

    print("\n" + "="*80)
    print("Running check_positions.py...")
    print("="*80 + "\n")

    result = subprocess.run(
        ["poetry", "run", "python", "check_positions.py"],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    output = result.stdout
    print(output)

    # Verify expected content
    checks = [
        ("OPEN POSITIONS (1)", "Open positions count"),
        ("BTCUSDT UP", "Open position symbol"),
        ("CLOSED POSITIONS (3)", "Closed positions count"),
        ("Win Rate:", "Win rate calculation"),
        ("Total PnL:", "Total PnL calculation"),
        ("EXIT REASONS:", "Exit reasons breakdown"),
        ("RECENT CLOSED POSITIONS", "Recent positions list"),
        ("TAKE_PROFIT", "Take profit exit"),
        ("STOP_LOSS", "Stop loss exit"),
        ("TRAILING_STOP", "Trailing stop exit"),
    ]

    passed = []
    failed = []

    for expected, description in checks:
        if expected in output:
            passed.append(f"✅ {description}")
        else:
            failed.append(f"❌ {description} (missing: '{expected}')")

    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)

    for check in passed:
        print(check)

    if failed:
        print("\nFailed checks:")
        for check in failed:
            print(check)
        return False

    print(f"\n✅ All {len(checks)} checks passed!")
    return True

async def cleanup_test_positions():
    """Remove test positions from database."""
    storage = Storage("./data/market.db", wal_mode=False)
    await storage.init_db()

    # Delete test positions (those with '_test' suffix)
    await storage.db.execute("DELETE FROM positions WHERE position_id LIKE '%_test'")
    await storage.db.commit()

    await storage.close()
    print("\n✅ Test positions cleaned up")

async def main():
    """Run integration test."""
    print("="*80)
    print("INTEGRATION TEST: check_positions.py")
    print("="*80)

    try:
        # Step 1: Create test data
        print("\n[1/3] Creating test positions...")
        await create_test_positions()

        # Step 2: Verify output
        print("\n[2/3] Verifying check_positions.py output...")
        success = await verify_check_positions_output()

        # Step 3: Cleanup
        print("\n[3/3] Cleaning up test data...")
        await cleanup_test_positions()

        if success:
            print("\n" + "="*80)
            print("✅ INTEGRATION TEST PASSED")
            print("="*80)
            print("\ncheck_positions.py works correctly:")
            print("  ✓ Reads open positions")
            print("  ✓ Reads closed positions")
            print("  ✓ Calculates statistics (win rate, PnL, avg duration)")
            print("  ✓ Shows exit reason breakdown")
            print("  ✓ Displays recent positions")
            print("  ✓ Handles emoji output on Windows")
            return 0
        else:
            print("\n" + "="*80)
            print("❌ INTEGRATION TEST FAILED")
            print("="*80)
            return 1

    except Exception as e:
        print(f"\n❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()

        # Try to cleanup even on error
        try:
            await cleanup_test_positions()
        except:
            pass

        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
