; RUN: opt < %s -print-controldep -analyze | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
define void @testembeddeddiamond() {
start:
 br i1 true, label %same, label %different
same: 
 br i1 true, label %samepart1, label %samepart2
samepart1:
 br label %samemergepoint
samepart2:
 br label %samemergepoint
samemergepoint:
 br label %returnit
different:
 br label %returnit
returnit:
    ret void
}
; CHECK: ({different},{samepart1},{samepart2},{returnit,start},{same,samemergepoint})
