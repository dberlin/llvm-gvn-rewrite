; RUN: opt -basicaa -memoryssa -dump-memoryssa -verify-memoryssa -disable-output < %s | FileCheck %s
; RUN: opt -basicaa -memoryssa -dump-memoryssa -verify-memoryssa -disable-output < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"

%struct.hoge = type { i32, %struct.widget }
%struct.widget = type { i64 }
define hidden void @quux() align 2 {
  %tmp = getelementptr inbounds %struct.hoge, %struct.hoge* undef, i64 0, i32 1, i32 0
  %tmp24 = getelementptr inbounds %struct.hoge, %struct.hoge* undef, i64 0, i32 1
  %tmp25 = bitcast %struct.widget* %tmp24 to i64**
  br label %bb26

bb26:                                             ; preds = %bb77, %0
; CHECK:  2 = MemoryPhi({%0,0},{bb77,3})
; 2 = MemoryPhi({%0,0},{bb77,3})
; CHECK-NEXT:   br i1 undef, label %bb68, label %bb77
  br i1 undef, label %bb68, label %bb77

bb68:                                             ; preds = %bb26
; CHECK:  MemoryUse(2)
; MemoryUse(2)
; CHECK-NEXT:   %tmp69 = load i64, i64* null, align 8
  %tmp69 = load i64, i64* null, align 8
; CHECK:  1 = MemoryDef(2)
; 1 = MemoryDef(2)
; CHECK-NEXT:   store i64 %tmp69, i64* %tmp, align 8
  store i64 %tmp69, i64* %tmp, align 8
  br label %bb77

bb77:                                             ; preds = %bb68, %bb26
; CHECK:  3 = MemoryPhi({bb68,1},{bb26,2})
; 3 = MemoryPhi({bb68,1},{bb26,2})
; CHECK:  MemoryUse(3)
; MemoryUse(3)
; CHECK-NEXT:   %tmp78 = load i64*, i64** %tmp25, align 8
  %tmp78 = load i64*, i64** %tmp25, align 8
  %tmp79 = getelementptr inbounds i64, i64* %tmp78, i64 undef
  br label %bb26
}
