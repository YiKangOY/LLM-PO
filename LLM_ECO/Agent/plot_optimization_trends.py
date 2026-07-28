#!/usr/bin/env python3
"""
Plot optimization trends across iterations
Visualizes timing (setup/hold), power, and area metrics
"""

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from typing import Dict, List
from configs import base_path
from eco_database import ReportParser


def collect_metrics_across_iterations(num_iterations: int) -> Dict[str, List[float]]:
    """
    Collect metrics across iterations by directly parsing report files

    Args:
        num_iterations: Number of iterations to process

    Returns:
        Dictionary containing lists of metrics for each iteration
    """
    metrics = {
        'iterations': [],
        # Setup timing
        'setup_tns': [],
        'setup_wns': [],
        'setup_violations': [],
        # Hold timing
        'hold_tns': [],
        'hold_wns': [],
        'hold_violations': [],
        # Power
        'total_power': [],
        'dynamic_power': [],
        'leakage_power': [],
        # Area
        'design_area': []
    }

    for iteration in range(num_iterations):
        print(f"Processing iteration {iteration}...")

        try:
            # Read QoR report for timing and area
            qor_file = os.path.join(base_path, f"reports/report_qor_{iteration}.txt")
            if not os.path.exists(qor_file):
                print(f"QoR report not found for iteration {iteration}, stopping")
                break

            with open(qor_file, 'r') as f:
                qor_content = f.read()

            qor_data = ReportParser.parse_qor_report(qor_content)

            # Read power report
            power_file = os.path.join(base_path, f"reports/report_power_{iteration}.txt")
            if not os.path.exists(power_file):
                print(f"Power report not found for iteration {iteration}, stopping")
                break

            with open(power_file, 'r') as f:
                power_content = f.read()

            power_data = ReportParser.parse_power_report(power_content)

            # Extract metrics
            metrics['iterations'].append(iteration)

            # Setup timing
            metrics['setup_violations'].append(qor_data['setup_violating_paths'])
            metrics['setup_wns'].append(qor_data['setup_critical_path_slack'])
            metrics['setup_tns'].append(qor_data['setup_total_negative_slack'])

            # Hold timing
            metrics['hold_violations'].append(qor_data['hold_violating_paths'])
            metrics['hold_wns'].append(qor_data['hold_critical_path_slack'])
            metrics['hold_tns'].append(qor_data['hold_total_negative_slack'])

            # Power (already in W, no conversion needed)
            total_power = power_data['total_power']
            leakage_power = power_data['leakage_power']
            dynamic_power = total_power - leakage_power

            metrics['total_power'].append(total_power)
            metrics['leakage_power'].append(leakage_power)
            metrics['dynamic_power'].append(dynamic_power)

            # Area
            metrics['design_area'].append(qor_data['design_area'])

        except Exception as e:
            print(f"Error processing iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            break

    return metrics


def plot_timing_trends(metrics: Dict[str, List[float]], output_file: str = 'timing_trends.png'):
    """
    Create a 4-subplot figure showing timing trends (separate plots for TNS and WNS)

    Args:
        metrics: Dictionary containing metric lists
        output_file: Output PNG filename
    """
    if not metrics['iterations']:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Timing Optimization Trends Across Iterations', fontsize=16, fontweight='bold')

    iterations = metrics['iterations']

    # Subplot 1: Setup WNS
    ax1 = axes[0, 0]
    ax1.plot(iterations, metrics['setup_wns'], 'r-o', label='Setup WNS', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Setup WNS (ns)', fontsize=11)
    ax1.set_title('Setup Worst Negative Slack', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Setup TNS
    ax2 = axes[0, 1]
    ax2.plot(iterations, metrics['setup_tns'], 'b-o', label='Setup TNS', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Setup TNS (ns)', fontsize=11)
    ax2.set_title('Setup Total Negative Slack', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Hold WNS
    ax3 = axes[1, 0]
    ax3.plot(iterations, metrics['hold_wns'], 'r-s', label='Hold WNS', linewidth=2, markersize=6)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Hold WNS (ns)', fontsize=11)
    ax3.set_title('Hold Worst Negative Slack', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Hold TNS
    ax4 = axes[1, 1]
    ax4.plot(iterations, metrics['hold_tns'], 'b-s', label='Hold TNS', linewidth=2, markersize=6)
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Hold TNS (ns)', fontsize=11)
    ax4.set_title('Hold Total Negative Slack', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    output_path = os.path.join(base_path, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Timing plot saved to: {output_path}")

    # Don't show the plot
    plt.close(fig)


def plot_power_area_trends(metrics: Dict[str, List[float]], output_file: str = 'power_area_trends.png'):
    """
    Create a 2-subplot figure showing power and area trends

    Args:
        metrics: Dictionary containing metric lists
        output_file: Output PNG filename
    """
    if not metrics['iterations']:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Power and Area Optimization Trends', fontsize=16, fontweight='bold')

    iterations = metrics['iterations']

    # Subplot 1: Power (Total, Dynamic, Leakage)
    ax1 = axes[0]
    ax1.plot(iterations, metrics['total_power'], 'g-o', label='Total Power', linewidth=2, markersize=6)
    ax1.plot(iterations, metrics['dynamic_power'], 'b-s', label='Dynamic Power', linewidth=2, markersize=6)
    ax1.plot(iterations, metrics['leakage_power'], 'r-^', label='Leakage Power', linewidth=2, markersize=6)

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Power (W)', fontsize=11)
    ax1.set_title('Power Components', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Area
    ax2 = axes[1]
    ax2.plot(iterations, metrics['design_area'], 'purple', marker='o', linewidth=2, markersize=6, label='Design Area')

    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Area (um²)', fontsize=11)
    ax2.set_title('Design Area', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    output_path = os.path.join(base_path, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Power/Area plot saved to: {output_path}")

    # Don't show the plot
    plt.close(fig)


def main():
    """Main function to collect metrics and create plots"""
    print("ECO Optimization Trends Plotting")
    print("=" * 50)

    # Determine number of iterations by checking for report files
    num_iterations = 0
    for i in range(100):  # Check up to 100 iterations
        qor_file = os.path.join(base_path, f"reports/report_qor_{i}.txt")
        if os.path.exists(qor_file):
            num_iterations = i + 1
        else:
            break

    if num_iterations == 0:
        print("No report files found in reports/ directory")
        return

    print(f"Found {num_iterations} iterations")

    # Collect metrics
    metrics = collect_metrics_across_iterations(num_iterations)

    if not metrics['iterations']:
        print("No metrics collected")
        return

    print(f"Collected metrics for {len(metrics['iterations'])} iterations")

    # Create plots
    plot_timing_trends(metrics)
    plot_power_area_trends(metrics)

    print("\nMetrics Summary:")
    print(f"Setup violations: {metrics['setup_violations'][0]} -> {metrics['setup_violations'][-1]}")
    print(f"Hold violations: {metrics['hold_violations'][0]} -> {metrics['hold_violations'][-1]}")
    print(f"Total power: {metrics['total_power'][0]:.6f} -> {metrics['total_power'][-1]:.6f} W")
    print(f"Design area: {metrics['design_area'][0]:.4f} -> {metrics['design_area'][-1]:.4f} um²")


if __name__ == "__main__":
    main()
