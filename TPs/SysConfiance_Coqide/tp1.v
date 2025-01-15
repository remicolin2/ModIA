(* Les règles de la déduction naturelle doivent être ajoutées à Coq. *) 
Require Import Naturelle. 

(* Ouverture d'une section *) 
Section LogiquePropositions. 

(* Déclaration de variables propositionnelles *) 
Variable A B C E Y R : Prop. 

Theorem Thm_0 : A /\ B -> B /\ A.
I_imp HAetB.
I_et.
E_et_d A.
Hyp HAetB.
E_et_g B.
Hyp HAetB.
Qed.

Theorem Thm_1 : ((A \/ B) -> C) -> (B -> C).
(* A COMPLETER *)
I_imp H1.
I_imp H2.

E_imp (A \/ B).
Hyp H1.
I_ou_d.

Hyp H2.
Qed.

Theorem Thm_2 : A -> ~~A.
(* A COMPLETER *)
I_imp H1.
I_non H.
E_non (A).
Hyp H1.
Hyp H.
Qed.

Theorem Thm_3 : (A -> B) -> (~B -> ~A).
(* A COMPLETER *)
I_imp H1.
I_imp H2.
I_non H3.
E_non B.
E_imp A.
Hyp H1.
Hyp H3.
Hyp H2.
Qed.

Theorem Thm_4 : (~~A) -> A.
(* A COMPLETER *)
I_imp H1.
absurde H.
E_non (~A).
Hyp H.
Hyp H1.
Qed.

Theorem Thm_5 : (~B -> ~A) -> (A -> B).
(* A COMPLETER *)
I_imp H1.
I_imp H2.
absurde H3.
E_non A.
Hyp H2.
E_imp (~B).
Hyp H1.
Hyp H3.
Qed.

Theorem Thm_6 : ((E -> (Y \/ R)) /\ (Y -> R)) -> ~R -> ~E.
(* A COMPLETER *)
I_imp H1.
I_imp H2.
I_non H3.
E_non R.
E_ou Y R.
E_imp E.
E_et_g (Y->R).
Hyp H1.
Hyp H3.
E_et_d (E-> Y\/R).
Hyp H1.
I_imp Hr.
Hyp Hr.
Hyp H2.
Qed.

(* Version en Coq *)

Theorem Coq_Thm_0 : A /\ B -> B /\ A.
intro H.
destruct H as (HA,HB).  (* élimination conjonction *)
split.                  (* introduction conjonction *)
exact HB.               (* hypothèse *)
exact HA.               (* hypothèse *)
Qed.


Theorem Coq_E_imp : ((A -> B) /\ A) -> B.
(* A COMPLETER *)
intro H.
destruct H as (HA,HB).
cut A.
Hyp HA.
Hyp HB.
Qed.

Theorem Coq_E_et_g : (A /\ B) -> A.
(* A COMPLETER *)
intro H.
destruct H as (HA,HB).
Hyp HA.
Qed.

Theorem Coq_E_ou : ((A \/ B) /\ (A -> C) /\ (B -> C)) -> C.
(* A COMPLETER *)
intro H.
destruct H as (HA,HB).
destruct HB as (HB1,HB2).
destruct HA as [HA1|HA2].
cut A.
exact HB1.
exact HA1.
cut B.
exact HB2.
exact HA2.
Qed.



Theorem Coq_Thm_7 : ((E -> (Y \/ R)) /\ (Y -> R)) -> (~R -> ~E).
(* A COMPLETER *)
intros.
intro H1.
destruct H as (H2, H3).
destruct (H2 H1) as [H4 | H5].
absurd (R).
exact H0.
cut Y.
exact H3.
exact H4.
absurd (R).
exact H0.
exact H5.
Qed.


End LogiquePropositions.

