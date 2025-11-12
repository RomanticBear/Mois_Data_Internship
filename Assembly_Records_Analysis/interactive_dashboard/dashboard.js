// 데이터 로드 및 대시보드 초기화
let analysisData = null;
let selectedParties = new Set();

// 데이터 로드
async function loadData() {
    try {
        // 먼저 같은 디렉토리의 data.json 시도
        let response = await fetch('data.json');
        if (!response.ok) {
            // 실패하면 상위 디렉토리에서 시도
            response = await fetch('../analysis_results/제415회_openai_analysis.json');
        }
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        analysisData = await response.json();
        initializeDashboard();
    } catch (error) {
        console.error('데이터 로드 실패:', error);
        document.querySelector('.container').innerHTML = 
            '<div class="loading"><h2>데이터를 로드할 수 없습니다.</h2><p>data.json 또는 ../analysis_results/제415회_openai_analysis.json 파일을 확인해주세요.</p><p>오류: ' + error.message + '</p></div>';
    }
}

// 대시보드 초기화
function initializeDashboard() {
    updateStats();
    renderKeyIssuesChart();
    renderPartyConcernsChart();
    renderQAQualityChart();
    renderPartyPositionHeatmap();
    renderPartyNetwork();
    setupPartyFilter();
}

// 통계 업데이트
function updateStats() {
    document.getElementById('total-speeches').textContent = 
        analysisData.total_speeches?.toLocaleString() || '-';
    document.getElementById('quality-speeches').textContent = 
        analysisData.quality_speeches?.toLocaleString() || '-';
    
    const keyIssuesCount = analysisData.session_summary?.key_issues?.length || 0;
    document.getElementById('key-issues-count').textContent = keyIssuesCount;
    
    const qaPairs = analysisData.qa_analysis?.qa_pairs_count || 
                   analysisData.qa_analysis?.total_qa_pairs || 0;
    document.getElementById('qa-pairs').textContent = qaPairs.toLocaleString();
}

// 핵심 이슈 중요도 차트
function renderKeyIssuesChart() {
    const container = d3.select('#key-issues-chart');
    container.selectAll('*').remove();

    const issues = analysisData.session_summary?.key_issues || [];
    if (issues.length === 0) return;

    const importanceMap = { '높음': 3, '중간': 2, '낮음': 1 };
    const data = issues.map(d => ({
        issue: d.issue,
        importance: importanceMap[d.importance] || 2,
        importanceLabel: d.importance,
        description: d.description,
        parties: d.mentioned_parties || []
    })).sort((a, b) => b.importance - a.importance);

    const width = container.node().clientWidth || 800;
    const height = Math.max(300, data.length * 60);
    const margin = { top: 20, right: 150, bottom: 40, left: 200 };

    const svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);

    const x = d3.scaleLinear()
        .domain([0, 4])
        .range([margin.left, width - margin.right]);

    const y = d3.scaleBand()
        .domain(data.map(d => d.issue))
        .range([margin.top, height - margin.bottom])
        .padding(0.2);

    // 막대
    const bars = svg.selectAll('.bar')
        .data(data)
        .enter().append('rect')
        .attr('class', 'bar')
        .attr('x', margin.left)
        .attr('y', d => y(d.issue))
        .attr('width', 0)
        .attr('height', y.bandwidth())
        .attr('fill', d => {
            const colors = { '높음': '#e74c3c', '중간': '#f39c12', '낮음': '#3498db' };
            return colors[d.importanceLabel] || '#95a5a6';
        })
        .attr('rx', 5)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 0.8);
            showTooltip(event, `
                <strong>${d.issue}</strong><br>
                중요도: ${d.importanceLabel}<br>
                ${d.description}<br>
                언급 정당: ${d.parties.join(', ')}
            `);
        })
        .on('mousemove', function(event) {
            moveTooltip(event);
        })
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 1);
            hideTooltip();
        });

    bars.transition()
        .duration(1000)
        .attr('width', d => x(d.importance) - margin.left);

    // 레이블
    svg.selectAll('.label')
        .data(data)
        .enter().append('text')
        .attr('class', 'label')
        .attr('x', d => x(d.importance) + 10)
        .attr('y', d => y(d.issue) + y.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('fill', '#333')
        .attr('font-size', '12px')
        .text(d => d.importanceLabel);

    // Y축
    svg.append('g')
        .attr('transform', `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(y))
        .selectAll('text')
        .attr('font-size', '13px')
        .style('text-anchor', 'end');

    // X축
    svg.append('g')
        .attr('transform', `translate(0, ${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(4))
        .append('text')
        .attr('x', width / 2)
        .attr('y', 35)
        .attr('fill', '#333')
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .text('중요도');
}

// 정당별 관심사 차트
function renderPartyConcernsChart() {
    const container = d3.select('#party-concerns-chart');
    container.selectAll('*').remove();

    const partyPositions = analysisData.session_summary?.party_positions || {};
    if (Object.keys(partyPositions).length === 0) return;

    const data = Object.entries(partyPositions).map(([party, info]) => ({
        party: party,
        concerns: info.main_concerns || [],
        concernsCount: (info.main_concerns || []).length,
        keyStatements: info.key_statements || '',
        stance: info.stance || ''
    })).filter(d => selectedParties.size === 0 || selectedParties.has(d.party))
       .sort((a, b) => b.concernsCount - a.concernsCount);

    if (data.length === 0) return;

    const width = container.node().clientWidth || 800;
    const height = Math.max(300, data.length * 80);
    const margin = { top: 20, right: 200, bottom: 40, left: 150 };

    const svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);

    const x = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.concernsCount)])
        .range([margin.left, width - margin.right]);

    const y = d3.scaleBand()
        .domain(data.map(d => d.party))
        .range([margin.top, height - margin.bottom])
        .padding(0.3);

    const colorScale = d3.scaleOrdinal()
        .domain(data.map(d => d.party))
        .range(['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b']);

    // 막대
    svg.selectAll('.bar')
        .data(data)
        .enter().append('rect')
        .attr('class', 'bar')
        .attr('x', margin.left)
        .attr('y', d => y(d.party))
        .attr('width', 0)
        .attr('height', y.bandwidth())
        .attr('fill', d => colorScale(d.party))
        .attr('rx', 5)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 0.8);
            showTooltip(event, `
                <strong>${d.party}</strong><br>
                주요 관심사 수: ${d.concernsCount}개<br>
                입장: ${d.stance}<br>
                주요 발언: ${d.keyStatements.substring(0, 100)}...
            `);
        })
        .on('mousemove', moveTooltip)
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 1);
            hideTooltip();
        })
        .transition()
        .duration(1000)
        .attr('width', d => x(d.concernsCount) - margin.left);

    // 관심사 리스트
    svg.selectAll('.concerns')
        .data(data)
        .enter().append('text')
        .attr('class', 'concerns')
        .attr('x', d => x(d.concernsCount) + 15)
        .attr('y', d => y(d.party) + y.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('fill', '#666')
        .attr('font-size', '11px')
        .text(d => d.concerns.join(', '));

    // Y축
    svg.append('g')
        .attr('transform', `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(y))
        .selectAll('text')
        .attr('font-size', '13px');

    // X축
    svg.append('g')
        .attr('transform', `translate(0, ${height - margin.bottom})`)
        .call(d3.axisBottom(x))
        .append('text')
        .attr('x', width / 2)
        .attr('y', 35)
        .attr('fill', '#333')
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .text('주요 관심사 수');
}

// 질의-응답 품질 차트
function renderQAQualityChart() {
    const container = d3.select('#qa-quality-chart');
    container.selectAll('*').remove();

    const qaAnalysis = analysisData.qa_analysis;
    if (!qaAnalysis) return;

    const width = container.node().clientWidth || 800;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };

    const svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);

    // 품질 분포 파이 차트
    const qualityData = qaAnalysis.quality_distribution || {};
    const qualityValues = [
        { label: '고품질', value: parseFloat(qualityData.high) || 0, color: '#2ecc71' },
        { label: '중품질', value: parseFloat(qualityData.medium) || 0, color: '#f39c12' },
        { label: '저품질', value: parseFloat(qualityData.low) || 0, color: '#e74c3c' }
    ];

    const pieWidth = 200;
    const pieHeight = 200;
    const pieX = margin.left + 100;
    const pieY = height / 2;

    const pie = d3.pie()
        .value(d => d.value)
        .sort(null);

    const arc = d3.arc()
        .innerRadius(0)
        .outerRadius(80);

    const arcs = svg.selectAll('.arc')
        .data(pie(qualityValues))
        .enter().append('g')
        .attr('class', 'arc')
        .attr('transform', `translate(${pieX}, ${pieY})`);

    arcs.append('path')
        .attr('d', arc)
        .attr('fill', d => d.data.color)
        .attr('stroke', 'white')
        .attr('stroke-width', 2)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 0.8);
            showTooltip(event, `<strong>${d.data.label}</strong><br>${d.data.value}%`);
        })
        .on('mousemove', moveTooltip)
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 1);
            hideTooltip();
        });

    // 레이블
    arcs.append('text')
        .attr('transform', d => `translate(${arc.centroid(d)})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(d => d.data.value > 5 ? `${d.data.value}%` : '');

    // 범례
    const legendX = pieX + 120;
    const legendY = pieY - 60;
    qualityValues.forEach((d, i) => {
        const g = svg.append('g')
            .attr('transform', `translate(${legendX}, ${legendY + i * 25})`);
        
        g.append('rect')
            .attr('width', 15)
            .attr('height', 15)
            .attr('fill', d.color);
        
        g.append('text')
            .attr('x', 20)
            .attr('y', 12)
            .attr('font-size', '12px')
            .text(`${d.label}: ${d.value}%`);
    });

    // 질문 유형 분포
    const questionTypes = qaAnalysis.question_types || {};
    const typeData = [
        { type: '정책 질의', value: parseFloat(questionTypes.policy_inquiry) || 0 },
        { type: '사실 확인', value: parseFloat(questionTypes.fact_checking) || 0 },
        { type: '비판 질의', value: parseFloat(questionTypes.criticism) || 0 },
        { type: '제안 질의', value: parseFloat(questionTypes.suggestion) || 0 }
    ];

    const barWidth = 300;
    const barHeight = 200;
    const barX = width - margin.right - barWidth;
    const barY = margin.top;

    const xType = d3.scaleBand()
        .domain(typeData.map(d => d.type))
        .range([0, barWidth])
        .padding(0.2);

    const yType = d3.scaleLinear()
        .domain([0, d3.max(typeData, d => d.value)])
        .range([barHeight, 0]);

    const typeBars = svg.selectAll('.type-bar')
        .data(typeData)
        .enter().append('g')
        .attr('transform', `translate(${barX}, ${barY})`);

    typeBars.append('rect')
        .attr('class', 'type-bar')
        .attr('x', d => xType(d.type))
        .attr('y', d => yType(d.value))
        .attr('width', xType.bandwidth())
        .attr('height', 0)
        .attr('fill', '#667eea')
        .attr('rx', 3)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 0.8);
            showTooltip(event, `<strong>${d.type}</strong><br>${d.value}%`);
        })
        .on('mousemove', moveTooltip)
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 1);
            hideTooltip();
        })
        .transition()
        .duration(1000)
        .attr('height', d => barHeight - yType(d.value));

    typeBars.append('text')
        .attr('x', d => xType(d.type) + xType.bandwidth() / 2)
        .attr('y', d => yType(d.value) - 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', '#333')
        .text(d => `${d.value}%`);

    // X축
    svg.append('g')
        .attr('transform', `translate(${barX}, ${barY + barHeight})`)
        .call(d3.axisBottom(xType))
        .selectAll('text')
        .attr('font-size', '11px')
        .attr('transform', 'rotate(-45) translate(-10, 5)');

    // Y축
    svg.append('g')
        .attr('transform', `translate(${barX}, ${barY})`)
        .call(d3.axisLeft(yType))
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -30)
        .attr('x', -barHeight / 2)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .text('비율 (%)');
}

// 안건별 정당 입장 히트맵
function renderPartyPositionHeatmap() {
    const container = d3.select('#party-position-heatmap');
    container.selectAll('*').remove();

    const partyPositions = analysisData.party_positions || {};
    const agendas = Object.keys(partyPositions);
    if (agendas.length === 0) return;

    // 모든 정당 수집
    const allParties = new Set();
    agendas.forEach(agenda => {
        const positions = partyPositions[agenda]?.party_positions || {};
        Object.keys(positions).forEach(party => allParties.add(party));
    });

    const parties = Array.from(allParties);
    const positionMap = { '지지': 2, '중립': 1, '반대': -1, '비판적': -2, '건의적': 1, '지지적': 2 };

    const data = [];
    agendas.forEach(agenda => {
        const positions = partyPositions[agenda]?.party_positions || {};
        parties.forEach(party => {
            const position = positions[party]?.position || positions[party]?.stance || '';
            data.push({
                agenda: agenda.length > 30 ? agenda.substring(0, 30) + '...' : agenda,
                party: party,
                position: position,
                value: positionMap[position] || 0
            });
        });
    });

    const width = container.node().clientWidth || 800;
    const height = Math.max(400, agendas.length * 50);
    const margin = { top: 100, right: 20, bottom: 40, left: 150 };

    const svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);

    const x = d3.scaleBand()
        .domain(parties)
        .range([margin.left, width - margin.right])
        .padding(0.1);

    const y = d3.scaleBand()
        .domain(agendas.map(a => a.length > 30 ? a.substring(0, 30) + '...' : a))
        .range([margin.top, height - margin.bottom])
        .padding(0.1);

    const colorScale = d3.scaleSequential()
        .domain([-2, 2])
        .interpolator(d3.interpolateRdBu);

    // 셀
    svg.selectAll('.cell')
        .data(data)
        .enter().append('rect')
        .attr('class', 'cell')
        .attr('x', d => x(d.party))
        .attr('y', d => y(d.agenda))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => colorScale(d.value))
        .attr('stroke', 'white')
        .attr('stroke-width', 1)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('stroke-width', 3);
            showTooltip(event, `<strong>${d.party}</strong><br>${d.agenda}<br>입장: ${d.position}`);
        })
        .on('mousemove', moveTooltip)
        .on('mouseout', function() {
            d3.select(this).attr('stroke-width', 1);
            hideTooltip();
        });

    // X축
    svg.append('g')
        .attr('transform', `translate(0, ${margin.top})`)
        .call(d3.axisTop(x))
        .selectAll('text')
        .attr('transform', 'rotate(-45)')
        .attr('text-anchor', 'end')
        .attr('font-size', '11px');

    // Y축
    svg.append('g')
        .attr('transform', `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(y))
        .selectAll('text')
        .attr('font-size', '10px');
}

// 정당 간 네트워크 그래프
function renderPartyNetwork() {
    const container = d3.select('#party-network');
    container.selectAll('*').remove();

    const conflicts = analysisData.session_summary?.major_conflicts || [];
    if (conflicts.length === 0) return;

    // 노드 생성
    const nodes = new Map();
    conflicts.forEach(conflict => {
        conflict.parties_involved.forEach(party => {
            if (!nodes.has(party)) {
                nodes.set(party, { id: party, group: 1 });
            }
        });
    });

    // 링크 생성
    const links = conflicts.map(conflict => ({
        source: conflict.parties_involved[0],
        target: conflict.parties_involved[1],
        nature: conflict.nature
    }));

    const width = container.node().clientWidth || 800;
    const height = 400;

    const svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);

    const simulation = d3.forceSimulation(Array.from(nodes.values()))
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));

    const link = svg.append('g')
        .selectAll('.link')
        .data(links)
        .enter().append('line')
        .attr('class', 'link')
        .attr('stroke', d => {
            const colors = { '대립': '#e74c3c', '토론': '#f39c12', '협력': '#2ecc71' };
            return colors[d.nature] || '#95a5a6';
        })
        .attr('stroke-width', 3)
        .attr('stroke-opacity', 0.6);

    const node = svg.append('g')
        .selectAll('.node')
        .data(Array.from(nodes.values()))
        .enter().append('circle')
        .attr('class', 'node')
        .attr('r', 20)
        .attr('fill', '#667eea')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended))
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 25);
            showTooltip(event, `<strong>${d.id}</strong>`);
        })
        .on('mousemove', moveTooltip)
        .on('mouseout', function() {
            d3.select(this).attr('r', 20);
            hideTooltip();
        });

    const label = svg.append('g')
        .selectAll('.label')
        .data(Array.from(nodes.values()))
        .enter().append('text')
        .attr('class', 'label')
        .attr('dx', 25)
        .attr('dy', 5)
        .attr('font-size', '12px')
        .attr('fill', '#333')
        .text(d => d.id);

    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);

        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// 정당 필터 설정
function setupPartyFilter() {
    const partyPositions = analysisData.session_summary?.party_positions || {};
    const parties = Object.keys(partyPositions);
    
    const filterContainer = d3.select('#party-filter');
    filterContainer.selectAll('*').remove();

    parties.forEach(party => {
        const label = filterContainer.append('label')
            .attr('class', 'party-checkbox')
            .text(party);

        label.append('input')
            .attr('type', 'checkbox')
            .attr('checked', true)
            .on('change', function() {
                if (this.checked) {
                    selectedParties.add(party);
                } else {
                    selectedParties.delete(party);
                }
                renderPartyConcernsChart();
            });

        selectedParties.add(party);
    });
}

// 툴팁 함수
function showTooltip(event, content) {
    const tooltip = d3.select('#tooltip');
    tooltip
        .html(content)
        .style('display', 'block')
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
}

function moveTooltip(event) {
    const tooltip = d3.select('#tooltip');
    tooltip
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
}

function hideTooltip() {
    d3.select('#tooltip').style('display', 'none');
}

// 페이지 로드 시 데이터 로드
window.addEventListener('DOMContentLoaded', loadData);

