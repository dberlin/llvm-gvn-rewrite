// RUN: llvm-profdata merge -o %t.profdata %S/Inputs/showProjectSummary.proftext

int main(int argc, char ** argv) {
  int x=0;
  for (int i = 0; i < 20; ++i)
    x *= 2;
  if (x >= 100)
    x = x / 2;
  else
    x = x * 2;
  return x;
}

// Test console output.
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile %t.profdata -filename-equivalence %s | FileCheck -check-prefixes=TEXT,TEXT-FILE,TEXT-HEADER %s
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile %t.profdata -project-title "Test Suite" -filename-equivalence %s | FileCheck -check-prefixes=TEXT-TITLE,TEXT,TEXT-FILE,TEXT-HEADER %s
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -instr-profile %t.profdata -project-title "Test Suite" -name=main -filename-equivalence %s | FileCheck -check-prefixes=TEXT-FUNCTION,TEXT-HEADER %s
// TEXT-TITLE: Test Suite
// TEXT: Coverage Report
// TEXT: Created:
// TEXT-FILE: showProjectSummary.cpp:
// TEXT-FILE: showProjectSummary.covmapping:
// TEXT-FUNCTION: main:

// Test html output.
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -format=html -o %t.dir -instr-profile %t.profdata -filename-equivalence %s
// RUN: FileCheck -check-prefixes=HTML,HTML-FILE,HTML-HEADER -input-file %t.dir/coverage/tmp/showProjectSummary.cpp.html %s
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -format=html -o %t.dir -instr-profile %t.profdata -project-title "Test Suite" -filename-equivalence %s
// RUN: FileCheck -check-prefixes=HTML-TITLE,HTML,HTML-FILE,HTML-HEADER -input-file %t.dir/coverage/tmp/showProjectSummary.cpp.html %s
// RUN: FileCheck -check-prefixes=HTML-TITLE,HTML -input-file %t.dir/index.html %s
// RUN: llvm-cov show %S/Inputs/showProjectSummary.covmapping -format=html -o %t.dir -instr-profile %t.profdata  -project-title "Test Suite" -filename-equivalence -name=main %s
// RUN: FileCheck -check-prefixes=HTML-FUNCTION,HTML-HEADER -input-file %t.dir/functions.html %s
// HTML-TITLE: <h1>Test Suite</h1>
// HTML: <h2>Coverage Report</h2>
// HTML: <h4>Created:{{.*}}</h4>
// HTML-FILE: <pre>{{.*}}showProjectSummary.cpp (Binary: showProjectSummary.covmapping)</pre>
// HTML-FUNCTION: <pre>main</pre>
// HTML-HEADER: <td><pre>Line No.</pre></td>
// HTML-HEADER: <td><pre>Count</pre></td>
// HTML-HEADER: <td><pre>Source (<a href='#L8'>jump to first uncovered line</a>)</pre></td>
