%define project_name qat-fermion

# Input
%{?!major:          %define major           0}
%{?!minor:          %define minor           0}
%{?!patchlevel:     %define patchlevel      0}
%{?!buildnumber:    %define buildnumber     0}
%{?!branch:         %define branch          master}

%if "%{branch}" == "rc" || "%{buildnumber}" == "0"
%{?!version:        %define version         %{major}.%{minor}.%{patchlevel}}
%else
%{?!version:        %define version         %{major}.%{minor}.%{patchlevel}.%{buildnumber}}
%endif

%{?!rpm_release:    %define rpm_release     bull.0.0}
%{?!python_major:   %define python_major    3}
%{?!python_minor:   %define python_minor    6}
%{?!packager:       %define packager        noreply@eviden.com}
%{?!run_by_jenkins: %define run_by_jenkins  0}
%{?!platform:       %define platform        linux-x86_64}
%{?!python_distrib: %define python_distrib  linux-x86_64}

# Defines
%define python_version      %{python_major}.%{python_minor}
%define python_rpm          python%{python_major}%{python_minor}
%define workspace           %{getenv:WORKSPACE}
%define project_prefix      %(echo %{project_name} | cut -d- -f1)
%define project_suffix      %(echo %{project_name} | cut -d- -f2-)

# Read location environment variables
%define target_bin_dir      /%{getenv:BIN_INSTALL_DIR}
%define target_lib_dir      /%{getenv:LIB_INSTALL_DIR}
%define target_headers_dir  /%{getenv:HEADERS_INSTALL_DIR}
%define target_thrift_dir   /%{getenv:THRIFT_INSTALL_DIR}
%define target_python_dir   /%{getenv:PYTHON_INSTALL_DIR}

%undefine __brp_mangle_shebangs

%if 0%{?srcrpm}
%undefine dist
%endif

# -------------------------------------------------------------------
#
# GLOBAL PACKAGE & SUB-PACKAGES DEFINITION
#
# -------------------------------------------------------------------
Name:           %{project_prefix}%{python_major}%{python_minor}-%{project_suffix}
Version:        %{version}
Release:        %{rpm_release}%{?dist}
Group:          Development/Libraries
Distribution:   QLM
Vendor:         Eviden
License:        Bull S.A.S. proprietary : All rights reserved
ExclusiveArch:  x86_64
URL:            https://eviden.com/solutions/advanced-computing/quantum-computing

Source:         %{project_name}-%{version}.tar.gz
Source1:        qat.tar.gz


# -------------------------------------------------------------------
#
# MAIN PACKAGE DEFINITION
#
# -------------------------------------------------------------------
Summary:  Quantum Application Toolset (QAT)
Provides: %{name}
AutoReq: no

%description
qat-fermion simulator. This package replaces the qat-dqs package.

# -------------------------------------------------------------------
#
# PREP
#
# -------------------------------------------------------------------
%prep
%setup -q -n %{project_name}-%{version}
tar xfz %{SOURCE1} -C ..


# -------------------------------------------------------------------
#
# BUILD
#
# -------------------------------------------------------------------
%build


# -------------------------------------------------------------------
#
# INSTALL
#
# -------------------------------------------------------------------
%install
mkdir -p $RPM_BUILD_ROOT
dist_no_dot=$(echo "%{dist}" | sed 's/^\.//')
tar xfz %{workspace}/artifacts/tarballs-prod/%{name}-%{version}-%{rpm_release}-${dist_no_dot}-%{_target_cpu}.tar.gz -C $RPM_BUILD_ROOT


# -------------------------------------------------------------------
#
# FILES
#
# -------------------------------------------------------------------
# Main package
%files
%defattr(-,root,root)
%{target_python_dir}/qat/*


# -------------------------------------------------------------------
#
# SCRIPTLETS
#
# -------------------------------------------------------------------
%pre

%post

%postun

%preun


# -------------------------------------------------------------------
#
# CHANGELOG
#
# -------------------------------------------------------------------
%changelog
* Sat May 04 2024 Jerome Pioux <jerome.pioux@eviden.com>
- Release 1.10
  Change files location to support Virtual Environments.
  Create a versionned main package using the python version, and
  an unversionned, noarch, config package.

-* Thu July 7 2022 Jerome Pioux <jerome.pioux@atos.net>
-- Initial release
