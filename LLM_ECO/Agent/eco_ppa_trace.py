
class TraceMemory:
    def __init__(self):
        self.traces = []

    def add_trace(self, results):
        new_trace = {
            'iterations': [],
            'setup_TNS_trace': [],
            'setup_WNS_trace': [],
            'hold_TNS_trace': [],
            'hold_WNS_trace': [],
            'area_trace': [],
            'total_power_trace': [],
            'dynamic_power_trace': [],
            'leakage_power_trace': [],
            'combinational_power_trace': [],
            'unfixable_reasons_trace': [],
            'execution_time_trace': [],
            'command_trace': []
        }
        for result in results:
            current_state = result['unified_analysis']['current_state']
            timing_state = current_state['timing']
            area_state = current_state['area']
            power_state = current_state['power']
            new_trace['iterations'].append(result['iteration'])
            new_trace['setup_TNS_trace'].append(timing_state['setup_total_negative_slack'])
            new_trace['setup_WNS_trace'].append(timing_state['setup_critical_path_slack'])
            new_trace['hold_TNS_trace'].append(timing_state['hold_total_negative_slack'])
            new_trace['hold_WNS_trace'].append(timing_state['hold_critical_path_slack'])
            new_trace['area_trace'].append(area_state['design_area'])
            new_trace['total_power_trace'].append(power_state['total_power'])
            new_trace['dynamic_power_trace'].append(power_state['dynamic_power'])
            new_trace['leakage_power_trace'].append(power_state['leakage_power'])
            new_trace['combinational_power_trace'].append(power_state['combinational_power'])
            new_trace['unfixable_reasons_trace'].append(result['unified_analysis']['unfixable_reasons'])
            new_trace['execution_time_trace'].append(result['execution_result']['execution_time'])
            new_trace['command_trace'].append(result['execution_result']['tcl_command'])
        self.traces.append(new_trace)

    def get_traces(self):
        return self.traces

    def build_trace_from_history(self, design_state_history, unfixable_analysis_history):
        trace = {
            'iterations': [],
            'setup_TNS_trace': [],
            'setup_WNS_trace': [],
            'hold_TNS_trace': [],
            'hold_WNS_trace': [],
            'area_trace': [],
            'total_power_trace': [],
            'dynamic_power_trace': [],
            'leakage_power_trace': [],
            'combinational_power_trace': [],
            'unfixable_reasons_trace': [],
            'execution_time_trace': [],
            'command_trace': []
        }

        for hist_entry in design_state_history:
            executed_command = hist_entry['executed_command']
            if executed_command == "":
                continue
            design_state = hist_entry['design_state']
            timing_state = design_state['timing']
            area_state = design_state['area']
            power_state = design_state['power']

            trace['iterations'].append(hist_entry['iteration'])
            trace['setup_TNS_trace'].append(timing_state['setup_total_negative_slack'])
            trace['setup_WNS_trace'].append(timing_state['setup_critical_path_slack'])
            trace['hold_TNS_trace'].append(timing_state['hold_total_negative_slack'])
            trace['hold_WNS_trace'].append(timing_state['hold_critical_path_slack'])
            trace['area_trace'].append(area_state['design_area'])
            trace['total_power_trace'].append(power_state['total_power'])
            trace['dynamic_power_trace'].append(power_state['dynamic_power'])
            trace['leakage_power_trace'].append(power_state['leakage_power'])
            trace['combinational_power_trace'].append(power_state['combinational_power'])
            trace['execution_time_trace'].append(hist_entry['actual_execution_time'])
            trace['command_trace'].append(executed_command)

            unfixable_reasons = {}
            for unfix_entry in unfixable_analysis_history:
                if unfix_entry['iteration'] == hist_entry['iteration']:
                    unfixable_reasons = unfix_entry['unfixable_reasons']
                    break
            trace['unfixable_reasons_trace'].append(unfixable_reasons)

        return trace

    def extract_pareto_traces(self):
        """Extract Pareto-optimal traces from stored trace memory."""
        candidates = []
        for trace in self.traces:
            final_metrics = self._final_metrics_from_trace(trace)
            candidates.append({
                "commands": trace['command_trace'],
                "design_trace": trace,
                "final_metrics": final_metrics
            })

        def dominates(metrics_a, metrics_b):
            better_or_equal = (
                metrics_a["setup_tns"] >= metrics_b["setup_tns"]
                and metrics_a["setup_wns"] >= metrics_b["setup_wns"]
                and metrics_a["hold_tns"] >= metrics_b["hold_tns"]
                and metrics_a["hold_wns"] >= metrics_b["hold_wns"]
                and metrics_a["total_power"] <= metrics_b["total_power"]
                and metrics_a["design_area"] <= metrics_b["design_area"]
            )
            strictly_better = (
                metrics_a["setup_tns"] > metrics_b["setup_tns"]
                or metrics_a["setup_wns"] > metrics_b["setup_wns"]
                or metrics_a["hold_tns"] > metrics_b["hold_tns"]
                or metrics_a["hold_wns"] > metrics_b["hold_wns"]
                or metrics_a["total_power"] < metrics_b["total_power"]
                or metrics_a["design_area"] < metrics_b["design_area"]
            )
            return better_or_equal and strictly_better

        pareto_traces = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if candidate is other:
                    continue
                if dominates(other["final_metrics"], candidate["final_metrics"]):
                    dominated = True
                    break
            if not dominated:
                pareto_traces.append((candidate["commands"], candidate["design_trace"]))

        return pareto_traces

    def _final_metrics_from_trace(self, trace):
        if not trace["iterations"]:
            raise ValueError("trace must include at least one iteration")
        last_idx = len(trace["iterations"]) - 1
        return {
            "setup_tns": trace["setup_TNS_trace"][last_idx],
            "setup_wns": trace["setup_WNS_trace"][last_idx],
            "hold_tns": trace["hold_TNS_trace"][last_idx],
            "hold_wns": trace["hold_WNS_trace"][last_idx],
            "total_power": trace["total_power_trace"][last_idx],
            "dynamic_power": trace["dynamic_power_trace"][last_idx],
            "leakage_power": trace["leakage_power_trace"][last_idx],
            "design_area": trace["area_trace"][last_idx]
        }
