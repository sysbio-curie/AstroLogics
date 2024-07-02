#!/usr/bin/env bash
clingo -W no-atom-undefined -t 1 0 --single-shot --project -c bounded_nonreach=0 "${@}" - <<EOF
#program base.
{clause(N,1..C,L,S): in(L,N,S), maxC(N,C), node(N), node(L)}.
:- clause(N,_,L,S), clause(N,_,L,-S).
1 { constant(N,(-1;1)) } 1 :- node(N), not clause(N,_,_,_).
constant(N) :- constant(N,_).
size(N,C,X) :- X = #count {L,S: clause(N,C,L,S)}; clause(N,C,_,_); maxC(N,_).
:- clause(N,C,_,_); not clause(N,C-1,_,_); C > 1; maxC(N,_).
:- size(N,C1,X1); size(N,C2,X2); X1 < X2; C1 > C2; maxC(N,_).
:- size(N,C1,X); size(N,C2,X); C1 > C2; mindiff(N,C1,C2,L1) ; mindiff(N,C2,C1,L2) ; L1 < L2; maxC(N,_).
clausediff(N,C1,C2,L) :- clause(N,C1,L,_);not clause(N,C2,L,_);clause(N,C2,_,_), C1 != C2; maxC(N,_).
mindiff(N,C1,C2,L) :- clausediff(N,C1,C2,L); L <= L' : clausediff(N,C1,C2,L'), clause(N,C1,L',_), C1!=C2; maxC(N,_).
:- size(N,C1,X1); size(N,C2,X2); C1 != C2; X1 <= X2; clause(N,C2,L,S) : clause(N,C1,L,S); maxC(N,_).
nbnode(12).
node("Pax6").
node("Hes5").
node("Mash1").
node("Scl").
node("Olig2").
node("Stat3").
node("Zic1").
node("Brn2").
node("Tuj1").
node("Myt1L").
node("Sox8").
node("Aldh1L1").
in("Pax6","Pax6",1).
in("Pax6","Hes5",1).
in("Pax6","Mash1",1).
in("Hes5","Mash1",-1).
in("Hes5","Scl",1).
in("Hes5","Olig2",1).
in("Hes5","Stat3",1).
in("Mash1","Hes5",-1).
in("Mash1","Zic1",1).
in("Mash1","Brn2",1).
in("Scl","Olig2",-1).
in("Scl","Stat3",1).
in("Olig2","Scl",-1).
in("Olig2","Myt1L",1).
in("Olig2","Sox8",1).
in("Olig2","Brn2",-1).
in("Stat3","Aldh1L1",1).
in("Zic1","Tuj1",1).
in("Brn2","Tuj1",1).
in("Myt1L","Tuj1",1).
maxC("Pax6",1).
maxC("Hes5",2).
maxC("Mash1",2).
maxC("Scl",2).
maxC("Olig2",2).
maxC("Stat3",2).
maxC("Zic1",1).
maxC("Brn2",2).
maxC("Tuj1",3).
maxC("Myt1L",1).
maxC("Sox8",1).
maxC("Aldh1L1",1).
obs("zero","Pax6",-1).
obs("zero","Hes5",-1).
obs("zero","Mash1",-1).
obs("zero","Scl",-1).
obs("zero","Olig2",-1).
obs("zero","Stat3",-1).
obs("zero","Zic1",-1).
obs("zero","Brn2",-1).
obs("zero","Tuj1",-1).
obs("zero","Myt1L",-1).
obs("zero","Sox8",-1).
obs("zero","Aldh1L1",-1).
obs("init","Pax6",1).
obs("init","Hes5",-1).
obs("init","Mash1",-1).
obs("init","Scl",-1).
obs("init","Olig2",-1).
obs("init","Stat3",-1).
obs("init","Zic1",-1).
obs("init","Brn2",-1).
obs("init","Tuj1",-1).
obs("init","Myt1L",-1).
obs("init","Sox8",-1).
obs("init","Aldh1L1",-1).
obs("tM","Pax6",1).
obs("tM","Tuj1",-1).
obs("tM","Scl",-1).
obs("tM","Aldh1L1",-1).
obs("tM","Olig2",-1).
obs("tM","Sox8",-1).
obs("fT","Pax6",1).
obs("fT","Tuj1",1).
obs("fT","Brn2",1).
obs("fT","Zic1",1).
obs("fT","Aldh1L1",-1).
obs("fT","Sox8",-1).
obs("tO","Pax6",1).
obs("tO","Tuj1",-1).
obs("tO","Scl",-1).
obs("tO","Aldh1L1",-1).
obs("tO","Olig2",1).
obs("tO","Sox8",-1).
obs("fMS","Pax6",1).
obs("fMS","Tuj1",-1).
obs("fMS","Zic1",-1).
obs("fMS","Brn2",-1).
obs("fMS","Aldh1L1",-1).
obs("fMS","Sox8",1).
obs("tS","Pax6",1).
obs("tS","Tuj1",-1).
obs("tS","Scl",1).
obs("tS","Aldh1L1",-1).
obs("tS","Olig2",-1).
obs("tS","Sox8",-1).
obs("fA","Pax6",1).
obs("fA","Tuj1",-1).
obs("fA","Zic1",-1).
obs("fA","Brn2",-1).
obs("fA","Aldh1L1",1).
obs("fA","Sox8",-1).
1 {cfg(X,N,(-1;1))} 1 :- cfg(X), node(N).
cfg(X,N,V) :- bind_cfg(X,O), obs(O,N,V), node(N).
eval(X,N,C,-1) :- clause(N,C,L,-V), mcfg(X,L,V), not clamped(X,N,_).
eval(X,N,C,1) :- mcfg(X,L,V): clause(N,C,L,V); clause(N,C,_,_), mcfg(X,_,_), not clamped(X,N,_).
eval(X,N,1) :- eval(X,N,C,1), clause(N,C,_,_).
eval(X,N,-1) :- eval(X,N,C,-1): clause(N,C,_,_); clause(N,_,_,_), mcfg(X,_,_).
eval(X,N,V) :- clamped(X,N,V).
eval(X,N,V) :- constant(N,V), mcfg(X,_,_), not clamped(X,N,_).
eval(X,N,V) :- evalbdd(X,N,V), node(N), not clamped(X,N,_).
evalbdd(X,V,V) :- mcfg(X,_,_), V=(-1;1).
evalbdd(X,B,V) :- bdd(B,N,_,HI), mcfg(X,N,1), evalbdd(X,HI,V).
evalbdd(X,B,V) :- bdd(B,N,LO,_), mcfg(X,N,-1), evalbdd(X,LO,V).
evalbdd(X,B,V) :- mcfg(X,_,_), bdd(B,V).
mcfg(X,N,V) :- ext(X,N,V).
cfg(__bocfg8,N,-1); cfg(__bocfg8,N,1) :- node(N).
cfg(__bocfg8,N,-V) :- cfg(__bocfg8,N,V), saturate(__bocfg8).
saturate(__bocfg8) :- valid(__bocfg8,Z): expect_valid(__bocfg8,Z).
:- not saturate(__bocfg8).
expect_valid(__bocfg8,__bocond4).
expect_valid(__bocfg8,__bocond5).
obs("fA").
cfg("fA").
bind_cfg("fA","fA").
mcfg(__bofp8,N,V) :- cfg("fA",N,V).
:- cfg("fA",N,V), eval(__bofp8,N,-V).
obs("fMS").
cfg("fMS").
bind_cfg("fMS","fMS").
mcfg(__bofp9,N,V) :- cfg("fMS",N,V).
:- cfg("fMS",N,V), eval(__bofp9,N,-V).
obs("fT").
cfg(("fT",0)).
bind_cfg(("fT",0),"fT").
mcfg(__bots4,N,V) :- cfg(("fT",0),N,V).
mcfg(__bots4,N,V) :- eval(__bots4,N,V).
:- obs("fT",N,V), mcfg(__bots4,N,-V).
obs("init").
cfg("init").
bind_cfg("init","init").
obs("tM").
cfg("tM").
bind_cfg("tM","tM").
mcfg(__boreach24,N,V) :- cfg("init",N,V).
ext(__boreach24,N,V) :- eval(__boreach24,N,V), cfg("tM",N,V).
:- cfg("tM",N,V), not mcfg(__boreach24,N,V).
:- cfg("tM",N,V), ext(__boreach24,N,-V), not ext(__boreach24,N,V).
{ext(__boreach24,N,V)} :- eval(__boreach24,N,V), cfg("tM",N,-V).
mcfg(__boreach25,N,V) :- cfg("tM",N,V).
ext(__boreach25,N,V) :- eval(__boreach25,N,V), cfg(("fT",0),N,V).
:- cfg(("fT",0),N,V), not mcfg(__boreach25,N,V).
:- cfg(("fT",0),N,V), ext(__boreach25,N,-V), not ext(__boreach25,N,V).
{ext(__boreach25,N,V)} :- eval(__boreach25,N,V), cfg(("fT",0),N,-V).
obs("tO").
cfg("tO").
bind_cfg("tO","tO").
mcfg(__boreach26,N,V) :- cfg("init",N,V).
ext(__boreach26,N,V) :- eval(__boreach26,N,V), cfg("tO",N,V).
:- cfg("tO",N,V), not mcfg(__boreach26,N,V).
:- cfg("tO",N,V), ext(__boreach26,N,-V), not ext(__boreach26,N,V).
{ext(__boreach26,N,V)} :- eval(__boreach26,N,V), cfg("tO",N,-V).
mcfg(__boreach27,N,V) :- cfg("tO",N,V).
ext(__boreach27,N,V) :- eval(__boreach27,N,V), cfg("fMS",N,V).
:- cfg("fMS",N,V), not mcfg(__boreach27,N,V).
:- cfg("fMS",N,V), ext(__boreach27,N,-V), not ext(__boreach27,N,V).
{ext(__boreach27,N,V)} :- eval(__boreach27,N,V), cfg("fMS",N,-V).
obs("tS").
cfg("tS").
bind_cfg("tS","tS").
mcfg(__boreach28,N,V) :- cfg("init",N,V).
ext(__boreach28,N,V) :- eval(__boreach28,N,V), cfg("tS",N,V).
:- cfg("tS",N,V), not mcfg(__boreach28,N,V).
:- cfg("tS",N,V), ext(__boreach28,N,-V), not ext(__boreach28,N,V).
{ext(__boreach28,N,V)} :- eval(__boreach28,N,V), cfg("tS",N,-V).
mcfg(__boreach29,N,V) :- cfg("tS",N,V).
ext(__boreach29,N,V) :- eval(__boreach29,N,V), cfg("fA",N,V).
:- cfg("fA",N,V), not mcfg(__boreach29,N,V).
:- cfg("fA",N,V), ext(__boreach29,N,-V), not ext(__boreach29,N,V).
{ext(__boreach29,N,V)} :- eval(__boreach29,N,V), cfg("fA",N,-V).
obs("zero").
cfg("zero").
bind_cfg("zero","zero").
mcfg((__bononreach12,1..K),N,V) :- reach_steps(__bononreach12,K), cfg("zero",N,V).
ext((__bononreach12,I),N,V) :- eval((__bononreach12,I),N,V), not locked((__bononreach12,I),N).
reach_bad(__bononreach12,I,N) :- cfg("zero",N,V), cfg(("fT",0),N,V), ext((__bononreach12,I),N,-V), not ext((__bononreach12,I),N,V).
locked((__bononreach12,I+1),N) :- reach_bad(__bononreach12,I,N), reach_steps(__bononreach12,K), I < K.
locked((__bononreach12,I+1),N) :- locked((__bononreach12,I),N), reach_steps(__bononreach12,K), I < K.
nr_ok(__bononreach12) :- reach_steps(__bononreach12,K), cfg(("fT",0),N,V), not mcfg((__bononreach12,K),N,V).
:- not nr_ok(__bononreach12).
reach_steps(__bononreach12,1).
mcfg((__bononreach13,1..K),N,V) :- reach_steps(__bononreach13,K), cfg("zero",N,V).
ext((__bononreach13,I),N,V) :- eval((__bononreach13,I),N,V), not locked((__bononreach13,I),N).
reach_bad(__bononreach13,I,N) :- cfg("zero",N,V), cfg("fMS",N,V), ext((__bononreach13,I),N,-V), not ext((__bononreach13,I),N,V).
locked((__bononreach13,I+1),N) :- reach_bad(__bononreach13,I,N), reach_steps(__bononreach13,K), I < K.
locked((__bononreach13,I+1),N) :- locked((__bononreach13,I),N), reach_steps(__bononreach13,K), I < K.
nr_ok(__bononreach13) :- reach_steps(__bononreach13,K), cfg("fMS",N,V), not mcfg((__bononreach13,K),N,V).
:- not nr_ok(__bononreach13).
reach_steps(__bononreach13,K) :- nbnode(K), bounded_nonreach <= 0.
reach_steps(__bononreach13,bounded_nonreach) :- bounded_nonreach > 0.
mcfg((__bononreach14,1..K),N,V) :- reach_steps(__bononreach14,K), cfg("zero",N,V).
ext((__bononreach14,I),N,V) :- eval((__bononreach14,I),N,V), not locked((__bononreach14,I),N).
reach_bad(__bononreach14,I,N) :- cfg("zero",N,V), cfg("fA",N,V), ext((__bononreach14,I),N,-V), not ext((__bononreach14,I),N,V).
locked((__bononreach14,I+1),N) :- reach_bad(__bononreach14,I,N), reach_steps(__bononreach14,K), I < K.
locked((__bononreach14,I+1),N) :- locked((__bononreach14,I),N), reach_steps(__bononreach14,K), I < K.
nr_ok(__bononreach14) :- reach_steps(__bononreach14,K), cfg("fA",N,V), not mcfg((__bononreach14,K),N,V).
:- not nr_ok(__bononreach14).
reach_steps(__bononreach14,K) :- nbnode(K), bounded_nonreach <= 0.
reach_steps(__bononreach14,bounded_nonreach) :- bounded_nonreach > 0.
mcfg(__bocfg9,N,V) :- cfg(__bocfg8,N,V).
valid(__bocfg8,__bocond4) :- cfg(__bocfg8,N,V), eval(__bocfg9,N,-V).
valid(__bocfg8,__bocond4) :- cfg(__bocfg8,N,V): obs("fA",N,V), node(N).
valid(__bocfg8,__bocond4) :- cfg(__bocfg8,N,V): obs("fMS",N,V), node(N).
valid(__bocfg8,__bocond4) :- cfg(__bocfg8,N,V): obs("zero",N,V), node(N).
valid(__bocfg8,__bocond4) :- cfg(__bocfg8,N,V): obs("fT",N,V), node(N).
mcfg(__bocfg10,N,V) :- cfg(__bocfg8,N,V).
valid(__bocfg8,__bocond5) :- cfg(__bocfg8,N,V), eval(__bocfg10,N,-V).
valid(__bocfg8,__bocond5) :- cfg(__bocfg8,N,V): obs("fA",N,V), node(N).
valid(__bocfg8,__bocond5) :- cfg(__bocfg8,N,V): obs("fMS",N,V), node(N).
valid(__bocfg8,__bocond5) :- cfg(__bocfg8,N,V): obs("fT",N,V), node(N).
mcfg(__bocfg11,N,V) :- cfg("init",N,V).
mcfg(__bocfg11,N,V) :- eval(__bocfg11,N,V).
valid(__bocfg8,__bocond5) :- cfg(__bocfg8,N,V), not mcfg(__bocfg11,N,V).
#show clause/4.
#show constant/2.

EOF
