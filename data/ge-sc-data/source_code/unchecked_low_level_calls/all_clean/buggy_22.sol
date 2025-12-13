pragma solidity >=0.5.1;
contract owned {
    address public owner;
    constructor() public {
        owner = msg.sender;
    }
function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address newOwner) onlyOwner public {
        owner = newOwner;
    }
function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
}
contract tokenRecipient {
  function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
  event receivedEther(address sender, uint amount);
  function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
  event receivedTokens(address _from, uint256 _value, address _token, bytes _extraData);
    function receiveApproval(address _from, uint256 _value, address _token, bytes memory _extraData) public {
        Token t = Token(_token);
        require(t.transferFrom(_from, address(this), _value));
        emit receivedTokens(_from, _value, _token, _extraData);
    }
function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
    function () payable external {
        emit receivedEther(msg.sender, msg.value);
    }
function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
}
contract Token {
    function totalSupply() public view returns (uint256);
function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
    function actualBalanceOf(address _owner) public view returns (uint256 balance);
function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
    function renounceOwnership() public;
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
    function transferOwnership(address _newOwner) public;
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function pause() public;
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    function unpause() public;
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
}
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "Safe mul error");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "Safe div error");
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "Safe sub error");
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "Safe add error");
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "Safe mod error");
        return a % b;
    }
}
contract MindsyncPlatform is owned, tokenRecipient {
    using SafeMath for uint256;
    uint public minimumQuorum;
    uint public minimumTokensToVote;
    uint public debatingPeriodInMinutes;
  function bug_unchk_send18() payable public{
      msg.sender.transfer(1 ether);}
  Proposal[] public proposals;
  function bug_unchk_send29() payable public{
      msg.sender.transfer(1 ether);}
  uint public numProposals;
  function bug_unchk_send6() payable public{
      msg.sender.transfer(1 ether);}
  Token public tokenAddress;
  function bug_unchk_send16() payable public{
      msg.sender.transfer(1 ether);}
  address chairmanAddress;
  function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
  bool public initialized = false;
  function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
  event Initialized();
  function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
  event ProposalAdded(uint proposalID, address recipient, uint amount, string description);
  function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
  event Voted(uint proposalID, bool position, address voter);
  function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event ProposalTallied(uint proposalID, uint result, uint quorum, bool active);
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event ChangeOfRules(uint newMinimumTokensToVote, uint newMinimumQuorum, uint newDebatingPeriodInMinutes, address newTokenAddress, address newChairmanAddress);
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event ProposalSignedByChairman(uint proposalNumber,  bool sign, address chairman);
    struct Proposal {
        address recipient;
        uint amount;
        string description;
        bool signedByChairman;
        uint minExecutionDate;
        bool executed;
        bool proposalPassed;
        uint numberOfVotes;
        bytes32 proposalHash;
        Vote[] votes;
        mapping (address => bool) voted;
    }
    struct Vote {
        bool inSupport;
        address voter;
    }
    modifier onlyTokenholders {
        require(tokenAddress.actualBalanceOf(msg.sender) > minimumTokensToVote);
        _;
    }
    modifier onlyChairman {
        require(msg.sender == chairmanAddress);
        _;
    }
    constructor() payable public {
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function init(Token _tokenAddress, address _chairmanAddress, uint _minimumTokensToVote, uint _minimumPercentToPassAVote, uint _minutesForDebate) onlyOwner public {
        require(!initialized);
        initialized = true;
        changeVotingRules(_tokenAddress, _chairmanAddress, _minimumTokensToVote, _minimumPercentToPassAVote, _minutesForDebate);
        emit Initialized();
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
    function changeVotingRules(Token _tokenAddress, address _chairmanAddress, uint _minimumTokensToVote, uint _minimumPercentToPassAVote, uint _minutesForDebate) onlyOwner public {
        require(_chairmanAddress != address(0));
        require(_minimumPercentToPassAVote <= 51);
        tokenAddress = Token(_tokenAddress);
        chairmanAddress = _chairmanAddress;
        if (_minimumTokensToVote == 0 ) _minimumTokensToVote = 1;
        minimumTokensToVote = _minimumTokensToVote;
        if (_minimumPercentToPassAVote == 0 ) _minimumPercentToPassAVote = 51;
        minimumQuorum = _minimumPercentToPassAVote;
        debatingPeriodInMinutes = _minutesForDebate;
        emit ChangeOfRules(_minimumTokensToVote, minimumQuorum, debatingPeriodInMinutes, address(tokenAddress), chairmanAddress);
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function newProposal(
        address destination,
        uint weiAmount,
        string memory transactionDescription,
        bytes memory transactionBytecode
    )
        onlyTokenholders public
        returns (uint proposalID)
    {
        proposalID = proposals.length++;
        Proposal storage p = proposals[proposalID];
        p.recipient = destination;
        p.signedByChairman = false;
        p.amount = weiAmount;
        p.description = transactionDescription;
        p.proposalHash = keccak256(abi.encodePacked(destination, weiAmount, transactionBytecode));
        p.minExecutionDate = now + debatingPeriodInMinutes * 1 minutes;
        p.executed = false;
        p.proposalPassed = false;
        p.numberOfVotes = 0;
        emit ProposalAdded(proposalID, destination, weiAmount, transactionDescription);
        numProposals = proposalID+1;
        return proposalID;
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
    function checkProposalCode(
        uint proposalNumber,
        address destination,
        uint weiAmount,
        bytes memory transactionBytecode
    )
        view public
        returns (bool codeChecksOut)
    {
        Proposal storage p = proposals[proposalNumber];
        return p.proposalHash == keccak256(abi.encodePacked(destination, weiAmount, transactionBytecode));
    }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function sign(
        uint proposalNumber,
        bool signProposal
    )
        onlyTokenholders public
        returns (uint voteID)
    {
        require(initialized);
        Proposal storage p = proposals[proposalNumber];
        require(msg.sender == chairmanAddress);
        require(signProposal == true);
        p.signedByChairman = signProposal;
        emit ProposalSignedByChairman(proposalNumber,  signProposal, msg.sender);
        return proposalNumber;
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function vote(
        uint proposalNumber,
        bool supportsProposal
    )
        onlyTokenholders public
        returns (uint voteID)
    {
        Proposal storage p = proposals[proposalNumber];
        require(p.voted[msg.sender] != true);
        voteID = p.votes.length++;
        p.votes[voteID] = Vote({inSupport: supportsProposal, voter: msg.sender});
        p.voted[msg.sender] = true;
        p.numberOfVotes = voteID +1;
        emit Voted(proposalNumber,  supportsProposal, msg.sender);
        return voteID;
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function executeProposal(uint proposalNumber, bytes memory transactionBytecode) public {
        Proposal storage p = proposals[proposalNumber];
        require(initialized);
        require(now > p.minExecutionDate                                             
            && !p.executed                                                          
            && p.proposalHash == keccak256(abi.encodePacked(p.recipient, p.amount, transactionBytecode))); 
        uint quorum = 0;
        uint yea = 0;
        uint nay = 0;
        for (uint i = 0; i <  p.votes.length; ++i) {
            Vote storage v = p.votes[i];
            uint voteWeight = tokenAddress.actualBalanceOf(v.voter);
            quorum += voteWeight;
            if (v.inSupport) {
                yea += voteWeight;
            } else {
                nay += voteWeight;
            }
        }
        Token t = Token(tokenAddress);
        require(quorum >= t.totalSupply().mul(minimumQuorum).div(100)); 
        if (yea > nay ) {
            p.executed = true;
            (bool success, ) = p.recipient.call.value(p.amount)(transactionBytecode);
            require(success);
            p.proposalPassed = true;
        } else {
            p.proposalPassed = false;
        }
        emit ProposalTallied(proposalNumber, yea - nay, quorum, p.proposalPassed);
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
}