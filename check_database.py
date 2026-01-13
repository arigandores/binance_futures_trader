"""Database diagnostic script to check for data and calculation issues."""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path


def check_database():
    """Check database for data and potential issues."""
    db_path = Path("./data/market.db")

    if not db_path.exists():
        print(f"[ERROR] Database not found at {db_path}")
        return

    print(f"[OK] Database found at {db_path}\n")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check bars_1m table
    print("=" * 80)
    print("BARS DATA (1-minute bars)")
    print("=" * 80)

    cursor.execute("SELECT COUNT(*) FROM bars_1m")
    bar_count = cursor.fetchone()[0]
    print(f"Total bars: {bar_count}")

    if bar_count == 0:
        print("[ERROR] No bars found in database! WebSocket may not be receiving data.")
        conn.close()
        return

    # Get bar counts by symbol
    cursor.execute("""
        SELECT symbol, COUNT(*), MIN(ts_minute), MAX(ts_minute)
        FROM bars_1m
        GROUP BY symbol
    """)

    print("\nBars by symbol:")
    print("-" * 80)
    print(f"{'Symbol':<15} {'Count':<10} {'First Bar':<25} {'Last Bar':<25}")
    print("-" * 80)

    for row in cursor.fetchall():
        symbol, count, min_ts, max_ts = row
        first_bar = datetime.fromtimestamp(min_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
        last_bar = datetime.fromtimestamp(max_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{symbol:<15} {count:<10} {first_bar:<25} {last_bar:<25}")

    # Check recent bars for data quality
    print("\n" + "=" * 80)
    print("RECENT BARS DATA QUALITY (Last 10 bars per symbol)")
    print("=" * 80)

    cursor.execute("SELECT DISTINCT symbol FROM bars_1m")
    symbols = [row[0] for row in cursor.fetchall()]

    for symbol in symbols:
        print(f"\n{symbol}:")
        print("-" * 80)
        cursor.execute("""
            SELECT ts_minute, c, vol, taker_buy, taker_sell
            FROM bars_1m
            WHERE symbol = ?
            ORDER BY ts_minute DESC
            LIMIT 10
        """, (symbol,))

        rows = cursor.fetchall()
        print(f"{'Time':<20} {'Close':<12} {'Volume':<12} {'TakerBuy':<12} {'TakerSell':<12} {'TakerBuy%':<12}")
        print("-" * 80)

        for row in rows:
            ts, close, vol, taker_buy, taker_sell = row
            time_str = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')
            total_taker = taker_buy + taker_sell
            taker_pct = (taker_buy / total_taker * 100) if total_taker > 0 else 0
            print(f"{time_str:<20} {close:<12.2f} {vol:<12.2f} {taker_buy:<12.2f} {taker_sell:<12.2f} {taker_pct:<12.1f}")

    # Check features table
    print("\n" + "=" * 80)
    print("FEATURES DATA")
    print("=" * 80)

    cursor.execute("SELECT COUNT(*) FROM features")
    feature_count = cursor.fetchone()[0]
    print(f"Total features: {feature_count}")

    if feature_count == 0:
        print("[ERROR] No features calculated! May not have enough bars (need 15+ bars per symbol).")
        conn.close()
        return

    # Get feature counts by symbol
    cursor.execute("""
        SELECT symbol, COUNT(*), MIN(ts_minute), MAX(ts_minute)
        FROM features
        GROUP BY symbol
    """)

    print("\nFeatures by symbol:")
    print("-" * 80)
    print(f"{'Symbol':<15} {'Count':<10} {'First Feature':<25} {'Last Feature':<25}")
    print("-" * 80)

    for row in cursor.fetchall():
        symbol, count, min_ts, max_ts = row
        first_feat = datetime.fromtimestamp(min_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
        last_feat = datetime.fromtimestamp(max_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{symbol:<15} {count:<10} {first_feat:<25} {last_feat:<25}")

    # Check recent features and z-scores
    print("\n" + "=" * 80)
    print("RECENT FEATURES & Z-SCORES (Last 20 features per symbol)")
    print("=" * 80)
    print("Looking for z_er_15m >= 3.0 AND z_vol_15m >= 3.0 AND taker_buy_share >= 0.65 or <= 0.35")
    print("=" * 80)

    for symbol in symbols:
        print(f"\n{symbol}:")
        print("-" * 120)
        cursor.execute("""
            SELECT ts_minute, z_er_15m, z_vol_15m, taker_buy_share_15m, beta, er_15m
            FROM features
            WHERE symbol = ?
            ORDER BY ts_minute DESC
            LIMIT 20
        """, (symbol,))

        rows = cursor.fetchall()
        print(f"{'Time':<20} {'z_ER':<10} {'z_VOL':<10} {'TakerShare':<12} {'Beta':<10} {'ER_15m':<12} {'Alert?':<10}")
        print("-" * 120)

        alert_potential_count = 0
        for row in rows:
            ts, z_er, z_vol, taker_share, beta, er = row
            time_str = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')

            # Check if would trigger alert
            alert_trigger = ""
            if z_er is not None and z_vol is not None and taker_share is not None:
                meets_z_er = abs(z_er) >= 3.0
                meets_z_vol = z_vol >= 3.0
                meets_taker = taker_share >= 0.65 or taker_share <= 0.35

                if meets_z_er and meets_z_vol and meets_taker:
                    alert_trigger = "*** YES!"
                    alert_potential_count += 1
                elif meets_z_er or meets_z_vol:
                    alert_trigger = "! Close"

            z_er_str = f"{z_er:.2f}" if z_er is not None else "None"
            z_vol_str = f"{z_vol:.2f}" if z_vol is not None else "None"
            taker_str = f"{taker_share:.3f}" if taker_share is not None else "None"
            beta_str = f"{beta:.3f}" if beta is not None else "None"
            er_str = f"{er:.6f}" if er is not None else "None"

            print(f"{time_str:<20} {z_er_str:<10} {z_vol_str:<10} {taker_str:<12} {beta_str:<10} {er_str:<12} {alert_trigger:<10}")

        print(f"\nPotential alerts for {symbol}: {alert_potential_count}")

    # Check highest z-scores
    print("\n" + "=" * 80)
    print("HIGHEST Z-SCORES (Top 10 by z_er_15m)")
    print("=" * 80)

    cursor.execute("""
        SELECT symbol, ts_minute, z_er_15m, z_vol_15m, taker_buy_share_15m
        FROM features
        WHERE z_er_15m IS NOT NULL
        ORDER BY ABS(z_er_15m) DESC
        LIMIT 10
    """)

    print(f"{'Symbol':<15} {'Time':<20} {'z_ER':<10} {'z_VOL':<10} {'TakerShare':<12}")
    print("-" * 80)

    for row in cursor.fetchall():
        symbol, ts, z_er, z_vol, taker_share = row
        time_str = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')
        z_vol_str = f"{z_vol:.2f}" if z_vol is not None else "None"
        taker_str = f"{taker_share:.3f}" if taker_share is not None else "None"
        print(f"{symbol:<15} {time_str:<20} {z_er:<10.2f} {z_vol_str:<10} {taker_str:<12}")

    # Check events table
    print("\n" + "=" * 80)
    print("EVENTS")
    print("=" * 80)

    cursor.execute("SELECT COUNT(*) FROM events")
    event_count = cursor.fetchone()[0]
    print(f"Total events: {event_count}")

    if event_count == 0:
        print("[ERROR] No events detected. This is why you have 0 alerts.")
        print("\nPossible reasons:")
        print("1. Z-scores not high enough (need z_er >= 3.0 AND z_vol >= 3.0)")
        print("2. Taker buy share not extreme enough (need >= 0.65 or <= 0.35)")
        print("3. Market not volatile enough during the 3-hour period")
        print("4. Cooldown preventing alerts (check cooldown_tracker table)")
    else:
        print("\nRecent events:")
        print("-" * 120)
        cursor.execute("""
            SELECT event_id, ts, initiator_symbol, direction, status, metrics_json
            FROM events
            ORDER BY ts DESC
            LIMIT 10
        """)

        print(f"{'Event ID':<30} {'Time':<20} {'Symbol':<10} {'Dir':<5} {'Status':<15}")
        print("-" * 120)

        for row in cursor.fetchall():
            event_id, ts, symbol, direction, status, metrics = row
            time_str = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{event_id:<30} {time_str:<20} {symbol:<10} {direction:<5} {status:<15}")

    # Check cooldown tracker
    print("\n" + "=" * 80)
    print("COOLDOWN TRACKER")
    print("=" * 80)

    cursor.execute("SELECT COUNT(*) FROM cooldown_tracker")
    cooldown_count = cursor.fetchone()[0]

    if cooldown_count > 0:
        print(f"Cooldown entries: {cooldown_count}\n")
        cursor.execute("""
            SELECT symbol, last_alert_ts, last_direction
            FROM cooldown_tracker
        """)

        print(f"{'Symbol':<15} {'Last Alert Time':<25} {'Direction':<10}")
        print("-" * 60)

        for row in cursor.fetchall():
            symbol, last_ts, direction = row
            time_str = datetime.fromtimestamp(last_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{symbol:<15} {time_str:<25} {direction:<10}")
    else:
        print("No cooldown entries (no alerts have been triggered)")

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY & DIAGNOSTICS")
    print("=" * 80)

    print(f"\n[OK] Database has {bar_count} bars across {len(symbols)} symbols")
    print(f"[OK] Features calculated for {feature_count} data points")
    print(f"{'[OK]' if event_count > 0 else '[ERROR]'} Events detected: {event_count}")

    # Check if we have enough recent data
    cursor.execute("SELECT MAX(ts_minute) FROM bars_1m")
    latest_bar_ts = cursor.fetchone()[0]
    if latest_bar_ts:
        latest_bar_time = datetime.fromtimestamp(latest_bar_ts / 1000)
        time_since_last = datetime.now() - latest_bar_time
        print(f"\nLast bar received: {latest_bar_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time since last bar: {time_since_last}")

        if time_since_last > timedelta(minutes=5):
            print("! WARNING: Last bar is more than 5 minutes old. WebSocket may be disconnected!")

    # Check data distribution
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if event_count == 0:
        print("\nNo alerts were generated because:")
        print("1. Check the z-scores in the features table above")
        print("2. Look for any rows where z_er >= 3.0 AND z_vol >= 3.0 simultaneously")
        print("3. Verify taker_buy_share is extreme (>= 0.65 or <= 0.35)")
        print("4. Consider lowering thresholds in config.yaml if market is not volatile:")
        print("   - excess_return_z_initiator: 3.0 → 2.5")
        print("   - volume_z_initiator: 3.0 → 2.5")
        print("   - taker_dominance_min: 0.65 → 0.60")

    conn.close()


if __name__ == "__main__":
    try:
        check_database()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
