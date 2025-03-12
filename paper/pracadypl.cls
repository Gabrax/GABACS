%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%Klasa do składu prac dyplomowych%%%%%%%%%%
%%%%%%%%%%%%%%%%%%WMiI UŁ%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\def\@baseclass{report}
\def\@rodzajpracy{\mgr}
\DeclareOption{magisterska}{\def\@rodzajpracy{\mgr}}
\DeclareOption{licencjacka}{\def\@rodzajpracy{\lic}}
\DeclareOption{inzynierska}{\def\@rodzajpracy{\inz}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\@baseclass}}
\PassOptionsToClass{a4paper,twoside,openright,12pt}{\@baseclass}
\ProcessOptions
%
\LoadClass{\@baseclass}
%
\textwidth\paperwidth
\advance\textwidth -55mm
\oddsidemargin-1in
\advance\oddsidemargin 30mm
\evensidemargin-1in
\advance\evensidemargin 25mm
\topmargin -1in
\advance\topmargin 2cm
\setlength\textheight{46\baselineskip}
\addtolength\textheight{\topskip}
\marginparwidth15mm
%
\renewcommand\maketitle{%
  \begin{titlepage}%
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \begin{center}
      \includegraphics[scale=0.25]{logo_math_ul_h_pl_rgb.png}\par
      %
      %\vspace{2mm plus 1fill} 
      {\Huge\@author\par}
      \vspace{4mm}
      {\large Nr albumu:~\@nralbumu\par}
      \vspace{15mm plus .1fill}
      {\Huge\@title\par}
      \vspace{15mm plus .1fill}
      {\Large Praca \@rodzajpracy\\na kierunku \@kierunek\par}
      \vspace{10mm plus 1.5fill}
      \begin{flushright}\large
        \begin{tabular}{l}
          \vspace{2mm}
          Praca wykonana pod kierunkiem\\
          \vspace{2mm}
          \@opiekun\\
          \vspace{2mm}
          \@katedra
        \end{tabular}
      \end{flushright}
      %\vspace{1cm plus 1fill}
      \vfill
      {\large Łódź, \@date\par}
    \end{center}
    \@thanks
  \end{titlepage}%
  \begin{titlepage}
    \c@page=2
    \large
    \null
    \vfill
%    \begin{center}
        {\em \@zakres}\par
        \vspace{3cm}
        {\large\bf \sk :}
        \@slkluczowe\par
        \vspace{0.5cm}
        {\large\bf \et :}
        \@tytulang\par
        \vspace{0.5cm}
        {\large\bf \kw :}
        \@keywords\par
        \vspace{0.5cm}
%    \end{center}
  \end{titlepage}
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty  \setcounter{footnote}{0}
}
%
\def\nralbumu#1{\gdef\@nralbumu{#1}}
\def\@nralbumu{???\ClassError{pracamgr}{Brak numeru albumu}\@ehc}
%
\def\kierunek#1{\gdef\@kierunek{#1}}
\def\@kierunek{???\ClassError{pracamgr}{Nie podano kierunku studiow}\@ehc}
%
\def\zakres#1{\gdef\@zakres{#1}}
\def\@zakres{}
%
\def\opiekun#1{\gdef\@opiekun{#1}}
\def\@opiekun{???\ClassError{pracamgr}{Brak danych opiekuna pracy}\@ehc}
%
\def\katedra#1{\gdef\@katedra{#1}}
\def\@katedra{???\ClassError{pracamgr}{Brak danych katedry opiekuna}\@ehc}
%
\def\slkluczowe#1{\gdef\@slkluczowe{#1}}
\def\@slkluczowe{???\ClassError{pracamgr}{Brak slow kluczowych}\@ehc}
%
\def\keywords#1{\gdef\@keywords{#1}}
\def\@keywords{???\ClassError{pracamgr}{Brak slow kluczowych po angielsku}\@ehc}
%
\def\tytulang#1{\gdef\@tytulang{#1}}
\def\@tytulang{???\ClassError{pracamgr}{Brak tytulu po angielsku}\@ehc}
%
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
    \hbox{}\thispagestyle{empty}\newpage\if@twocolumn\hbox{}\newpage\fi\fi\fi}
%
\renewcommand*\@seccntformat[1]{\csname the#1\endcsname.\enspace}
\def\numberline#1{\hb@xt@\@tempdima{#1.\hfil}}
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\mdseries
      \leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
%
\endinput